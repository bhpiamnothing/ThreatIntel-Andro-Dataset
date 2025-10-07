import argparse
import json
import os
import re
import sys
import tempfile
from typing import List, Dict, Tuple
from multiprocessing import Pool

# ---------- Prevent numerical libraries from competing for CPU ----------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---- Third-party deps (only androguard and tqdm, optional) ----
try:
    from androguard.misc import AnalyzeAPK
    HAVE_ANDROGUARD = True
except Exception:
    print("[CRITICAL] Androguard not installed. pip install androguard", file=sys.stderr)
    HAVE_ANDROGUARD = False

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

# ----------------- Temp dir -----------------
TMP_DIR = os.path.join(tempfile.gettempdir(), "drebin_tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# ----------------- Suspicious names -----------------
SuspiciousNames = [
    "getExternalStorageDirectory",
    "getSimCountryIso",
    "execHttpRequest",
    "sendTextMessage",
    "getPackageInfo",
    "getSystemService",
    "setWifiDisabled",
    "Cipher",
    "Ljava/net/HttpURLconnection;->setRequestMethod(Ljava/lang/String;)",
    "Ljava/io/IOException;->printStackTrace",
    "Ljava/lang/Runtime;->exec",
    "system/bin/su",
]
SuspiciousNames_lc = {s.lower() for s in SuspiciousNames}

# ----------------- Axplorer mapping (only) -----------------
class AxplorerMapping(object):
    """
    Read axplorerPermApiAllUnified.json (permission → SDK → API list)
    Memory structure: self.api2perm_by_sdk = { sdk_int(str): { (Class+Method).lower(): permission } }
    """
    def __init__(self, axplorer_json_path="permission_api/axplorerPermApiAllUnified.json"):
        p = os.path.abspath(axplorer_json_path)
        if not os.path.exists(p):
            raise FileNotFoundError(f"axplorer json not found: {p}")

        with open(p, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)

        self.api2perm_by_sdk: Dict[str, Dict[str, str]] = {}
        for perm, sdk_map in data.items():
            if not isinstance(sdk_map, dict):
                continue
            for sdk_str, api_list in sdk_map.items():
                mp = self.api2perm_by_sdk.setdefault(str(sdk_str), {})
                if not isinstance(api_list, (list, tuple)):
                    continue
                for item in api_list:
                    # item: [Class, Method, Ret, [Args...]]
                    if not (isinstance(item, list) and len(item) >= 2):
                        continue
                    cls, mth = item[0], item[1]
                    if not (isinstance(cls, str) and isinstance(mth, str)):
                        continue
                    api_key = (cls + mth).lower()
                    mp[api_key] = perm  # later writes will overwrite

        self.numeric_sdks = sorted(int(k) for k in self.api2perm_by_sdk.keys() if k.isdigit())
        self.union_api2perm: Dict[str, str] = {}
        for _sdk, mp in self.api2perm_by_sdk.items():
            for k, v in mp.items():
                self.union_api2perm.setdefault(k, v)

        self.current_target_sdk = 9999  # default large value, use floor

    def set_target_sdk(self, target_i: int):
        try:
            self.current_target_sdk = int(target_i)
        except Exception:
            self.current_target_sdk = 9999

    def _select_mp_for_current_sdk(self) -> Dict[str, str]:
        key = str(self.current_target_sdk)
        if key in self.api2perm_by_sdk:
            return self.api2perm_by_sdk[key]
        floors = [s for s in self.numeric_sdks if s <= self.current_target_sdk]
        for s in reversed(floors):
            mp = self.api2perm_by_sdk.get(str(s))
            if mp:
                return mp
        return self.union_api2perm

    def GetPermFromApi(self, ApiClass: str, ApiMethodName: str):
        api_key = (str(ApiClass) + str(ApiMethodName)).lower()
        mp = self._select_mp_for_current_sdk()
        return mp.get(api_key, None)

# ----------------- Feature extraction -----------------
def extract_manifest_features_from_a(a) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Extract S1–S4 directly from the 'a' object"""
    requested_permission_list = list(a.get_permissions() or [])
    activity_list = list(a.get_activities() or [])
    service_list = list(a.get_services() or [])
    content_provider_list = list(a.get_providers() or [])
    broadcast_receiver_list = list(a.get_receivers() or [])
    hardware_list = list(a.get_features() or [])

    # intent actions
    intentfilter_list = []
    try:
        for comp_type, getter in [
            ('activity', a.get_activities() or []),
            ('service', a.get_services() or []),
            ('receiver', a.get_receivers() or []),
        ]:
            for comp in getter:
                try:
                    filters = a.get_intent_filters(comp_type, comp) or {}
                    for action in filters.get('action', []):
                        if action:
                            intentfilter_list.append(action)
                except Exception:
                    continue
    except Exception:
        pass

    # Remove duplicates and keep order
    def dedup_keep_order(x):
        seen = set(); out=[]
        for i in x:
            if i not in seen:
                out.append(i); seen.add(i)
        return out

    return (
        dedup_keep_order(requested_permission_list),
        dedup_keep_order(activity_list),
        dedup_keep_order(service_list),
        dedup_keep_order(content_provider_list),
        dedup_keep_order(broadcast_receiver_list),
        dedup_keep_order(hardware_list),
        dedup_keep_order(intentfilter_list),
    )

def extract_dex_features_from_a(a, dd, dx, pmap: AxplorerMapping, requested_permission_list: List[str]):
    """
    S5/S6: dx.get_external_methods()
    S7: Suspicious API hits
    S8: String pool priority, fallback to bytecode scan
    """
    used_permission_list, restricted_api_list, suspicious_api_list, url_list = [], [], [], []

    # 1) Align with targetSdk
    try:
        target = a.get_target_sdk_version()
        pmap.set_target_sdk(int(target) if target is not None else 9999)
    except Exception:
        pmap.set_target_sdk(9999)

    requested_set = set(requested_permission_list)

    # 2) S5/S6: External methods
    try:
        for m in dx.get_external_methods():
            cname = getattr(m, "class_name", None) or getattr(m, "get_class_name", lambda: None)()
            mname = getattr(m, "name", None) or getattr(m, "get_name", lambda: None)()
            if not cname or not mname:
                try:
                    mm = m.get_method()
                    cname = cname or getattr(mm, "get_class_name", lambda: None)()
                    mname = mname or getattr(mm, "get_name", lambda: None)()
                except Exception:
                    pass
            if not cname or not mname:
                continue

            api_class_dot = cname.replace('/', '.').replace('Landroid', 'android').strip(';')
            api_info = f"{api_class_dot}.{mname}"

            # S7
            if mname.lower() in SuspiciousNames_lc:
                suspicious_api_list.append(api_info)

            # S5/S6
            perm = pmap.GetPermFromApi(api_class_dot, mname)
            if perm is not None:
                if (len(requested_set) == 0) or (perm in requested_set):
                    used_permission_list.append(perm)
                    if api_info.lower() not in {s.lower() for s in suspicious_api_list}:
                        restricted_api_list.append(api_info)
                else:
                    if api_info.lower() not in {s.lower() for s in suspicious_api_list}:
                        restricted_api_list.append(api_info)
    except Exception:
        pass

    # 3) S8: String pool
    url_re = re.compile(r'http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F]{2}))+', re.IGNORECASE)
    def _iter_vms(dobj):
        if isinstance(dobj, (list, tuple)):
            for vm in dobj:
                yield vm
        elif dobj is not None:
            yield dobj

    for vm in _iter_vms(dd):
        try:
            for s in vm.get_strings():
                v = s.get_value() if hasattr(s, 'get_value') else str(s)
                if not v:
                    continue
                m = url_re.search(v)
                if m:
                    url = m.group()
                    url_domain = re.sub(r'(.*://)?([^/?]+).*', r'\g<1>\g<2>', url)
                    url_list.append(url_domain)
        except Exception:
            continue

    # 4) Fallback: Bytecode scan
    try:
        for vm in _iter_vms(dd):
            for meth in vm.get_methods():
                code = meth.get_code()
                if code is None:
                    continue
                bc = code.get_bc()
                for insn in bc.get_instructions():
                    line = f"{insn.get_name()} {insn.get_output()}"
                    lc = line.lower()
                    for s in SuspiciousNames_lc:
                        if s in lc:
                            suspicious_api_list.append(s)
                    m = url_re.search(line)
                    if m:
                        url = m.group()
                        url_domain = re.sub(r'(.*://)?([^/?]+).*', r'\g<1>\g<2>', url)
                        url_list.append(url_domain)
    except Exception:
        pass

    # Remove duplicates and sort
    used_permission_list = sorted(set(used_permission_list))
    restricted_api_list = sorted(set(restricted_api_list))
    suspicious_api_list = sorted(set(suspicious_api_list))
    url_list = sorted(set(url_list))
    return used_permission_list, restricted_api_list, suspicious_api_list, url_list

# ----------------- Subprocess worker -----------------
_GLOBAL_PMAP = None
_GLOBAL_OUTDIR = None

def _init_worker(axplorer_json_path: str, out_dir: str):
    """Initialize each subprocess once: load Axplorer mapping, prepare output directory"""
    global _GLOBAL_PMAP, _GLOBAL_OUTDIR
    _GLOBAL_PMAP = AxplorerMapping(axplorer_json_path=axplorer_json_path)
    _GLOBAL_OUTDIR = out_dir
    os.makedirs(_GLOBAL_OUTDIR, exist_ok=True)

def _process_one(apk_path: str):
    """
    Subprocess execution:
    - AnalyzeAPK
    - S1–S8 extraction
    - Write .data (key_value per line)
    - Return (ok, filepath, msg)
    """
    try:
        a, dd, dx = AnalyzeAPK(apk_path)
        if a is None or dx is None:
            return (False, apk_path, "AnalyzeAPK returned None")

        # Manifest
        (req_perms, acts, srvs, provs, rcvs, hw, intents) = extract_manifest_features_from_a(a)
        # Dex
        (used_perms, restricted_apis, suspicious_apis, urls) = extract_dex_features_from_a(
            a, dd, dx, _GLOBAL_PMAP, req_perms
        )

        # Write .data
        base = os.path.splitext(os.path.basename(apk_path))[0]
        data_path = os.path.join(_GLOBAL_OUTDIR, base + ".data")
        data_lines = {
            'requested_permission_list': req_perms,
            'activity_list': acts,
            'service_list': srvs,
            'content_provider_list': provs,
            'broadcast_receiver_list': rcvs,
            'hardware_list': hw,
            'intentfilter_list': intents,
            'used_permission_list': used_perms,
            'restricted_api_list': restricted_apis,
            'suspicious_api_list': suspicious_apis,
            'url_list': urls,
        }
        with open(data_path, 'w', encoding='utf-8') as f:
            for k, vs in data_lines.items():
                for v in vs:
                    f.write(f"{k}_{v}\n")

        return (True, apk_path, data_path)
    except Exception as e:
        return (False, apk_path, str(e))

# ----------------- Utils -----------------
def collect_apk_files(apk_dir: str) -> List[str]:
    out = []
    for root, _, files in os.walk(apk_dir):
        for fn in files:
            if fn.lower().endswith(".apk"):
                out.append(os.path.join(root, fn))
    out.sort()
    return out

def process_chunk_with_pool(apk_chunk: List[str], processes: int,
                            axplorer_json_path: str, out_dir: str, chunksize: int = 8):
    results = []
    with Pool(processes=processes, initializer=_init_worker,
              initargs=(axplorer_json_path, out_dir)) as pool:
        for ok, path, payload in tqdm(
            pool.imap_unordered(_process_one, apk_chunk, chunksize=chunksize),
            total=len(apk_chunk),
        ):
            results.append((ok, path, payload))
    return results

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="Drebin-style extractor (axplorer-only) — output .data only")
    ap.add_argument("--apk-dir", required=True, help="Directory of APKs (recursively)")
    ap.add_argument("--axplorer-json", default="permission_api/axplorerPermApiAllUnified.json",
                    help="Path to axplorer permission→API mapping JSON")
    ap.add_argument("--out-dir", required=True, help="Where to write per-apk .data files (key_value per line)")
    ap.add_argument("--processes", type=int, default=4, help="Number of worker processes")
    ap.add_argument("--chunk-size", type=int, default=500, help="How many APKs per Pool")
    ap.add_argument("--chunksize", type=int, default=8, help="imap_unordered chunk size per submit")
    args = ap.parse_args()

    if not HAVE_ANDROGUARD:
        sys.exit(2)

    apks = collect_apk_files(args.apk_dir)
    total = len(apks)
    if total == 0:
        print(f"[CRITICAL] No .apk files in {args.apk_dir}", file=sys.stderr)
        sys.exit(2)

    ok_cnt, fail_cnt = 0, 0
    print(f"[*] Found {total} APKs. Start extraction with {args.processes} processes ...")

    for i in range(0, total, args.chunk_size):
        chunk = apks[i : i + args.chunk_size]
        chunk_idx = i // args.chunk_size + 1
        total_chunks = (total + args.chunk_size - 1) // args.chunk_size
        print(f"\n--- Processing chunk {chunk_idx}/{total_chunks} ({len(chunk)} files) ---")

        results = process_chunk_with_pool(
            apk_chunk=chunk,
            processes=args.processes,
            axplorer_json_path=args.axplorer_json,
            out_dir=args.out_dir,
            chunksize=args.chunksize,
        )

        for ok, path, payload in results:
            if ok:
                ok_cnt += 1
            else:
                fail_cnt += 1
                print(f"[WARN] Fail: {path} -> {payload}", file=sys.stderr)

    print(f"[*] Done. Success: {ok_cnt}, Fail: {fail_cnt}")
    print(f"[*] .data files are in: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
