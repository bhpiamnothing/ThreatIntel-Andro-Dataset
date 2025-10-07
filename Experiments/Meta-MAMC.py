import os
import sys
import math
import time
import json
import argparse
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import csv

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------
# Utilities
# ---------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device(arg: str = "auto"):
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sparse_rows_to_dense(X_csr: csr_matrix, rows: np.ndarray) -> np.ndarray:
    sub = X_csr[rows]
    return sub.toarray().astype(np.float32)

def flatten_params(model: nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([p.view(-1) for p in model.parameters()])

def add_label_noise(y_onehot, noise_rate=0.1, seed=42):
    """
    Adds noise to training labels by flipping them in a one-hot matrix.
    y_onehot: numpy array, shape = (n_samples, n_classes), one-hot encoded matrix
    noise_rate: The proportion of labels to flip (0-1)
    """
    rng = np.random.default_rng(seed)
    y_noisy = y_onehot.copy()

    n_samples, n_classes = y_onehot.shape
    n_noisy = int(n_samples * noise_rate)

    if n_noisy == 0:
        return y_noisy

    noisy_indices = rng.choice(n_samples, size=n_noisy, replace=False)

    for idx in noisy_indices:
        current_label_arr = np.where(y_noisy[idx] == 1)[0]
        if len(current_label_arr) == 0:
            continue
        current_label = current_label_arr[0]
        
        possible_new_labels = list(range(n_classes))
        possible_new_labels.remove(current_label)
        
        if not possible_new_labels:
            continue

        new_label = rng.choice(possible_new_labels)
        
        y_noisy[idx, current_label] = 0
        y_noisy[idx, new_label] = 1

    return y_noisy

# ---------------------
# Train: MAML + Sampling + Evaluation
# ---------------------

@dataclass
class PerFamilyMetrics:
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    total_tn: int = 0
    acc: float = 0.0
    prec: float = 0.0
    rec: float = 0.0
    f1: float = 0.0

def functional_forward(params: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    x = F.linear(x, params[0], params[1])
    x = F.relu(x)
    x = F.linear(x, params[2], params[3])
    x = F.relu(x)
    x = F.linear(x, params[4], params[5])
    return x

def inner_update_params(params: List[torch.Tensor], x: torch.Tensor, y: torch.Tensor, lr: float, steps: int) -> List[torch.Tensor]:
    for _ in range(steps):
        pred = functional_forward(params, x)
        loss = F.cross_entropy(pred, y)
        grad = torch.autograd.grad(loss, params, create_graph=True)
        params = [p - lr * g for p, g in zip(params, grad)]
    return params

def family_based_sample(class_to_idx: Dict[int, np.ndarray], n_way: int, k_shot: int, q_query: int, X_csr: csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_classes_with_enough = [c for c, idx in class_to_idx.items() if len(idx) >= k_shot + q_query]
    if len(all_classes_with_enough) < n_way:
        all_classes_with_enough = [c for c, idx in class_to_idx.items() if len(idx) >= k_shot + 1]
        if not all_classes_with_enough:
            return np.array([]), np.array([]), np.array([]), np.array([])

    n_way_effective = min(n_way, len(all_classes_with_enough))
    if n_way_effective == 0:
         return np.array([]), np.array([]), np.array([]), np.array([])

    selected_classes = np.random.choice(all_classes_with_enough, n_way_effective, replace=False)
    
    support_x, support_y = [], []
    query_x, query_y = [], []
    
    for cls in selected_classes:
        cls_indices = class_to_idx[cls].copy()
        np.random.shuffle(cls_indices)
        
        actual_k_shot = min(k_shot, len(cls_indices) - 1)
        if actual_k_shot <= 0:
            continue
        actual_q_query = min(q_query, len(cls_indices) - actual_k_shot)
        
        if actual_q_query <= 0:
            continue

        support_idx = cls_indices[:actual_k_shot]
        query_idx = cls_indices[actual_k_shot : actual_k_shot + actual_q_query]
        
        support_x.extend(support_idx)
        support_y.extend([cls] * len(support_idx))
        query_x.extend(query_idx)
        query_y.extend([cls] * len(query_idx))
    
    support_x_arr = sparse_rows_to_dense(X_csr, np.array(support_x, dtype=int)) if support_x else np.array([])
    query_x_arr = sparse_rows_to_dense(X_csr, np.array(query_x, dtype=int)) if query_x else np.array([])
    return support_x_arr, np.array(support_y), query_x_arr, np.array(query_y)


def application_based_sample(class_to_idx: Dict[int, np.ndarray], n_way: int, k_shot: int, q_query: int, X_csr: csr_matrix, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(len(y))
    np.random.shuffle(all_indices)
    selected_indices = all_indices[:n_way * (k_shot + q_query)]
    
    selected_y = y[selected_indices]
    unique_classes = np.unique(selected_y)
    if len(unique_classes) < n_way:
        additional_classes = np.setdiff1d(np.unique(y), unique_classes)
        np.random.shuffle(additional_classes)
        additional_classes = additional_classes[:n_way - len(unique_classes)]
        for cls in additional_classes:
            if cls in class_to_idx and len(class_to_idx[cls]) > 0:
                cls_idx = np.random.choice(class_to_idx[cls], 1)[0]
                selected_indices = np.append(selected_indices, cls_idx)
    
    support_x, support_y = [], []
    query_x, query_y = [], []
    
    class_indices = {cls: [] for cls in np.unique(y[selected_indices]) if cls in class_to_idx}
    for i in selected_indices:
        cls = y[i]
        if cls in class_indices:
            class_indices[cls].append(i)
    
    selected_classes = list(class_indices.keys())[:n_way] 
    
    for cls in selected_classes:
        cls_idx = class_indices[cls]
        np.random.shuffle(cls_idx)
        support_num = min(k_shot, len(cls_idx) // 2)
        if support_num == 0 and len(cls_idx) > 1:
            support_num = 1
        query_num = min(q_query, len(cls_idx) - support_num)
        if support_num == 0 or query_num == 0:
            continue
        support_x.extend(cls_idx[:support_num])
        support_y.extend([cls] * support_num)
        query_x.extend(cls_idx[support_num:support_num + query_num])
        query_y.extend([cls] * query_num)
    
    support_x_arr = sparse_rows_to_dense(X_csr, np.array(support_x, dtype=int)) if support_x else np.array([])
    query_x_arr = sparse_rows_to_dense(X_csr, np.array(query_x, dtype=int)) if query_x else np.array([])
    return support_x_arr, np.array(support_y), query_x_arr, np.array(query_y)

def sample_task(
    class_to_idx: Dict[int, np.ndarray], n_way: int, k_shot: int, q_query: int, p: float,
    X_csr: csr_matrix, y: np.ndarray, num_query_only_classes: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if np.random.rand() < p:
        support_x, support_y, query_x, query_y = application_based_sample(class_to_idx, n_way, k_shot, q_query, X_csr, y)
    else:
        support_x, support_y, query_x, query_y = family_based_sample(class_to_idx, n_way, k_shot, q_query, X_csr)
    
    if num_query_only_classes > 0 and (support_y.size > 0 or query_y.size > 0):
        all_classes = list(class_to_idx.keys())
        used_classes = np.unique(np.concatenate((support_y, query_y)))
        available_classes = np.setdiff1d(all_classes, used_classes)
        if len(available_classes) >= num_query_only_classes:
            query_only_classes = np.random.choice(available_classes, num_query_only_classes, replace=False)
            for cls in query_only_classes:
                if cls in class_to_idx and len(class_to_idx[cls]) > 0:
                    cls_indices = class_to_idx[cls].copy()
                    np.random.shuffle(cls_indices)
                    query_idx = cls_indices[:q_query]
                    
                    if len(query_idx) > 0:
                        new_query_x = sparse_rows_to_dense(X_csr, query_idx)
                        query_x = np.concatenate((query_x, new_query_x)) if query_x.size > 0 else new_query_x

                        new_query_y = np.array([cls] * len(query_idx))
                        query_y = np.concatenate((query_y, new_query_y)) if query_y.size > 0 else new_query_y
    
    return support_x, support_y, query_x, query_y

def run_train(
    features_npz: str, labels_csv: str,
    global_train_ratio: float, meta_train_ratio: float,
    label_noise_ratio: float,
    n_way: int, k_shot: int, q_query: int, p: float,
    num_query_only_classes: int,
    inner_steps: int, inner_lr: float, meta_lr: float,
    meta_epochs: int, eps_stop: float, stop_patience: int,
    ft_epochs: int, ft_lr: float, batch_size: int,
    device_str: str, repeats: int, seed: int,
    save_per_family_csv: str, save_confusion_csv: str,
    size_threshold: int,
):
    set_seed(seed)
    device = pick_device(device_str)
    print(f"[*] Device: {device}")

    X_csr = load_npz(features_npz)
    labels_df = pd.read_csv(labels_csv)
    le = LabelEncoder()
    y_all = le.fit_transform(labels_df["family"].values)
    classes = le.classes_
    num_classes = len(classes)
    print(f"[*] Loaded {X_csr.shape[0]} samples, {num_classes} families")

    all_per_family: Dict[str, List[PerFamilyMetrics]] = {c: [] for c in classes}
    all_confusions: List[np.ndarray] = []
    all_acc: List[float] = []
    all_f1: List[float] = []
    all_rec: List[float] = []

    for rep in range(1, repeats + 1):
        print(f"\n--- Repeat {rep}/{repeats} ---")
        
        # CORRECT ORDER STEP 1: Get initial indices from the full dataset.
        sss_global = StratifiedShuffleSplit(n_splits=1, train_size=global_train_ratio, random_state=seed + rep)
        train_idx, test_idx = next(sss_global.split(np.zeros(len(y_all)), y_all))
        
        # CORRECT ORDER STEP 2: Modify the index lists *before* creating any data arrays.
        y_train_temp = y_all[train_idx]
        train_class_to_idx_temp = {c: np.flatnonzero(y_train_temp == c) for c in np.unique(y_train_temp)}
        
        y_test_temp = y_all[test_idx]
        test_class_to_idx_temp = {c: np.flatnonzero(y_test_temp == c) for c in np.unique(y_test_temp)}

        to_move_globals = []
        for c in list(train_class_to_idx_temp.keys()):
            if len(train_class_to_idx_temp[c]) < 2 and c in test_class_to_idx_temp and len(test_class_to_idx_temp[c]) >= 1:
                local_move_idx = test_class_to_idx_temp[c][0]
                global_idx_to_move = test_idx[local_move_idx]
                to_move_globals.append((c, global_idx_to_move))

        for c, global_idx in to_move_globals:
            train_idx = np.append(train_idx, global_idx)
            test_idx = test_idx[test_idx != global_idx]
            print(f"[*] Moved 1 sample of class {c} from test to train (global idx {global_idx})")

        # CORRECT ORDER STEP 3: Create the final train/test arrays using the finalized indices.
        # X_train_csr, y_train = X_csr[train_idx], y_all[train_idx]
        # X_test_csr, y_test = X_csr[test_idx], y_all[test_idx]
        # 加载原始标签（用于测试集）
        labels_df_orig = pd.read_csv(labels_csv)  # 包含 sha256 和 family

        # 加载训练集标签（来自 Kaspersky）
        train_labels_df = pd.read_csv("mydataset_kaspersky_results_1_aligned_mapped.csv")  # sha256, family

        # 创建原始标签映射（用于测试集）
        sha256_to_label_orig = labels_df_orig.set_index("sha256")["family"].to_dict()

        # 创建训练集标签映射（来自 Kaspersky）
        sha256_to_label_train = train_labels_df.set_index("sha256")["family"].to_dict()

        # 获取训练集和测试集的 sha256
        train_sha256 = labels_df_orig.iloc[train_idx]["sha256"].values
        test_sha256 = labels_df_orig.iloc[test_idx]["sha256"].values

        # 替换训练集标签（来自 Kaspersky）
        y_train = np.array([sha256_to_label_train.get(s, sha256_to_label_orig[s]) for s in train_sha256])
        y_test = np.array([sha256_to_label_orig[s] for s in test_sha256])

        # 重新编码标签（确保一致性）
        all_labels = np.concatenate([y_train, y_test])
        le = LabelEncoder()
        le.fit(all_labels)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
        num_classes = len(le.classes_)


        # 提取特征
        X_train_csr = X_csr[train_idx]
        X_test_csr = X_csr[test_idx]

        # ---------- 过滤训练集中样本数 < 2 的类 ----------
        from collections import Counter
        cls_cnt = Counter(y_train)  # 统计每个类有多少样本
        keep_cls = [c for c, n in cls_cnt.items() if n >= 2]
        mask = np.isin(y_train, keep_cls)  # 保留样本 mask
        train_idx = train_idx[mask]  # 同步裁剪索引
        y_train = y_train[mask]  # 同步裁剪标签
        X_train_csr = X_train_csr[mask]  # 同步裁剪特征
        print(f"[*] Filtered train: {len(y_train)} samples, {len(keep_cls)} classes")
        # --------------------------------------------------


        # CORRECT ORDER STEP 4: Inject noise into the correctly-sized `y_train` array.
        if label_noise_ratio > 0.0:
            print(f"[*] Injecting noise using one-hot logic at {label_noise_ratio:.2%} ratio.")
            rng = np.random.default_rng(seed + rep)

            # --- MODIFICATION START ---
            # Identify which samples are "safe" to corrupt.
            # A sample is safe if its original class has more than 2 members.
            class_counts = np.bincount(y_train, minlength=num_classes)
            safe_to_corrupt_indices = [
                i for i, label in enumerate(y_train) if class_counts[label] > 2
            ]

            num_to_corrupt = int(len(y_train) * label_noise_ratio)
            
            # We can only corrupt as many samples as are safe.
            if len(safe_to_corrupt_indices) > 0:
                actual_num_to_corrupt = min(num_to_corrupt, len(safe_to_corrupt_indices))
                
                if actual_num_to_corrupt > 0:
                    corrupt_indices = rng.choice(safe_to_corrupt_indices, size=actual_num_to_corrupt, replace=False)
                    
                    y_train_noisy = y_train.copy()
                    for idx in corrupt_indices:
                        original_label = y_train_noisy[idx]
                        possible_new_labels = list(range(num_classes))
                        possible_new_labels.remove(original_label)
                        
                        if possible_new_labels:
                            y_train_noisy[idx] = rng.choice(possible_new_labels)
                    
                    y_train = y_train_noisy
                    print(f"[*] Flipped {len(corrupt_indices)} labels safely.")
                else:
                    print("[*] No safe samples to corrupt, skipping noise injection.")
            else:
                print("[*] No classes with >2 samples, skipping noise injection.")
            # --- MODIFICATION END ---
        
        # Now proceed with the (potentially noisy) training data
        sss_meta = StratifiedShuffleSplit(n_splits=1, train_size=meta_train_ratio, random_state=seed + rep)
        meta_train_idx, meta_test_idx = next(sss_meta.split(np.zeros(len(y_train)), y_train))

        X_meta_train_csr, y_meta_train = X_train_csr[meta_train_idx], y_train[meta_train_idx]
        X_meta_test_csr, y_meta_test = X_train_csr[meta_test_idx], y_train[meta_test_idx]

        meta_train_class_to_idx = {c: np.flatnonzero(y_meta_train == c) for c in np.unique(y_meta_train)}
        
        input_dim = X_csr.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

        prev_loss = float('inf')
        patience_cnt = 0

        for epoch in range(1, meta_epochs + 1):
            model.train()
            meta_loss = 0.0
            num_valid_tasks = 0
            if (n_way * (k_shot + q_query)) > 0:
                num_tasks = max(1, len(y_meta_train) // (n_way * (k_shot + q_query)))
            else:
                num_tasks = 0

            for _ in range(num_tasks):
                support_x, support_y, query_x, query_y = sample_task(
                    meta_train_class_to_idx, n_way, k_shot, q_query, p, X_meta_train_csr, y_meta_train, num_query_only_classes=0
                )
                if len(support_y) == 0 or len(query_y) == 0:
                    continue
                support_x = torch.from_numpy(support_x).to(device)
                support_y = torch.from_numpy(support_y).long().to(device)
                query_x = torch.from_numpy(query_x).to(device)
                query_y = torch.from_numpy(query_y).long().to(device)

                base_params = [p.clone().detach().requires_grad_(True) for p in model.parameters()]
                fast_params = inner_update_params(base_params, support_x, support_y, inner_lr, inner_steps)

                query_pred = functional_forward(fast_params, query_x)
                task_loss = F.cross_entropy(query_pred, query_y)
                if torch.isnan(task_loss) or torch.isinf(task_loss):
                    continue
                meta_loss += task_loss
                num_valid_tasks += 1

            if num_valid_tasks > 0:
                meta_loss /= num_valid_tasks
            else:
                meta_loss = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            if meta_loss.requires_grad:
                meta_loss.backward()
                optimizer.step()

            print(f"[*] Meta Epoch {epoch}/{meta_epochs}: Loss {meta_loss.item():.4f}")

            if abs(prev_loss - meta_loss.item()) < eps_stop:
                patience_cnt += 1
                if patience_cnt >= stop_patience:
                    print(f"[*] Early stop at epoch {epoch}")
                    break
            else:
                patience_cnt = 0
            prev_loss = meta_loss.item()

        model.train()
        ft_optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr)
        ft_dataset = TensorDataset(
            torch.from_numpy(sparse_rows_to_dense(X_meta_test_csr, np.arange(X_meta_test_csr.shape[0]))),
            torch.from_numpy(y_meta_test).long()
        )
        ft_loader = DataLoader(ft_dataset, batch_size=batch_size, shuffle=True)

        for ft_epoch in range(1, ft_epochs + 1):
            ft_loss = 0.0
            for batch_x, batch_y in ft_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                ft_pred = model(batch_x)
                loss = F.cross_entropy(ft_pred, batch_y)
                ft_optimizer.zero_grad()
                loss.backward()
                ft_optimizer.step()
                ft_loss += loss.item()
            print(f"[*] FT Epoch {ft_epoch}/{ft_epochs}: Loss {ft_loss / len(ft_loader):.4f}")

        model.eval()
        with torch.no_grad():
            test_x = torch.from_numpy(sparse_rows_to_dense(X_test_csr, np.arange(X_test_csr.shape[0]))).to(device)
            test_pred = model(test_x).argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, test_pred, average='macro', zero_division=0)
        print(f"[*] Test: Acc {acc:.4f}, F1 {f1:.4f}, Recall {rec:.4f}")
        
        all_acc.append(acc)
        all_f1.append(f1)
        all_rec.append(rec)

        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y_test, test_pred):
            confusion[true, pred] += 1

        all_confusions.append(confusion)

        for c_idx, c_name in enumerate(classes):
            tp = confusion[c_idx, c_idx]
            fp = confusion[:, c_idx].sum() - tp
            fn = confusion[c_idx, :].sum() - tp
            tn = confusion.sum() - tp - fp - fn

            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1_c = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            acc_c = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0.0

            all_per_family[c_name].append(PerFamilyMetrics(
                total_tp=tp, total_fp=fp, total_fn=fn, total_tn=tn,
                acc=acc_c, prec=prec, rec=rec, f1=f1_c
            ))

    print("\n===== Summary over repeats =====")
    print(f"Acc mean±std: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Macro-F1 mean±std: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Macro-Recall mean±std: {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print("(Tip) For per-family metrics CSV, pass --save_per_family_csv; for confusion matrix CSV, pass --save_confusion_csv.")
    
    if save_per_family_csv:
        with open(save_per_family_csv, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["Family", "Size", "Avg_Acc", "Avg_Prec", "Avg_Rec", "Avg_F1"])
            for c_name, metrics_list in sorted(all_per_family.items(), key=lambda x: len(x[1]), reverse=True):
                if not metrics_list: continue
                size = sum(m.total_tp + m.total_fn for m in metrics_list) // repeats if repeats > 0 else 0
                avg_acc = np.mean([m.acc for m in metrics_list])
                avg_prec = np.mean([m.prec for m in metrics_list])
                avg_rec = np.mean([m.rec for m in metrics_list])
                avg_f1 = np.mean([m.f1 for m in metrics_list])
                writer.writerow([c_name, size, f"{avg_acc:.4f}", f"{avg_prec:.4f}", f"{avg_rec:.4f}", f"{avg_f1:.4f}"])

    if save_confusion_csv:
        avg_confusion = np.mean(all_confusions, axis=0)
        df_conf = pd.DataFrame(avg_confusion, index=classes, columns=classes)
        df_conf.to_csv(save_confusion_csv)

# ---------------------
# CLI
# ---------------------

def cli():
    ap = argparse.ArgumentParser(description="Meta-MAMC Trainer")
    
    # Training arguments
    ap.add_argument("--features_npz", default="drebin_res_features.npz", help="Path to the input feature matrix (.npz).")
    ap.add_argument("--labels_csv", default="labels_aligned.csv", help="Path to the input labels CSV file.")
    ap.add_argument("--global_train_ratio", type=float, default=0.7)
    ap.add_argument("--meta_train_ratio", type=float, default=0.7)
    ap.add_argument("--label_noise_ratio", type=float, default=0, help="Ratio of training labels to flip randomly. Default is 0.0 (no noise).")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--q_query", type=int, default=10)
    ap.add_argument("--p", type=float, default=0.25)
    ap.add_argument("--num_query_only_classes", type=int, default=1)
    ap.add_argument("--inner_steps", type=int, default=1)
    ap.add_argument("--inner_lr", type=float, default=0.01)
    ap.add_argument("--meta_lr", type=float, default=1e-3)
    ap.add_argument("--meta_epochs", type=int, default=30)
    ap.add_argument("--eps_stop", type=float, default=0.01)
    ap.add_argument("--stop_patience", type=int, default=3)
    ap.add_argument("--ft_epochs", type=int, default=15)
    ap.add_argument("--ft_lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=5, help="Global split repeats (paper: five-fold)")
    ap.add_argument("--save_per_family_csv", default="", help="Path to save per-family Precision/Recall/F1")
    ap.add_argument("--save_confusion_csv", default="", help="Path to save confusion matrix (labels x labels)")
    ap.add_argument("--size_threshold", type=int, default=10, help=">9=large families per paper")
    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()

# =========================================================
# Main
# =========================================================

def main():
    args = cli()
    run_train(
        features_npz=args.features_npz, labels_csv=args.labels_csv,
        global_train_ratio=args.global_train_ratio, meta_train_ratio=args.meta_train_ratio,
        label_noise_ratio=args.label_noise_ratio,
        n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query, p=args.p,
        num_query_only_classes=args.num_query_only_classes,
        inner_steps=args.inner_steps, inner_lr=args.inner_lr, meta_lr=args.meta_lr,
        meta_epochs=args.meta_epochs, eps_stop=args.eps_stop, stop_patience=args.stop_patience,
        ft_epochs=args.ft_epochs, ft_lr=args.ft_lr, batch_size=args.batch_size,
        device_str=args.device, repeats=args.repeats, seed=args.seed,
        save_per_family_csv=args.save_per_family_csv, save_confusion_csv=args.save_confusion_csv,
        size_threshold=args.size_threshold,
    )

if __name__ == "__main__":
    main()