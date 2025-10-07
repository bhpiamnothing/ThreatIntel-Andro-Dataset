#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM on Top-1000 RF-importance features
Inputs:
  - drebin_res_features.npz
  - labels.csv (must contain "family")
Output:
  - 5-repeat summary identical to the original script
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score

# ---------- config ----------
RANDOM_SEED = 42
REPEATS = 5
GLOBAL_TRAIN_RATIO = 0.7          # Same as the original script
RF_RANDOM_STATE = 42              # Fixed random seed for forest
TOPK = 1000                       # Select 1000 features
SVM_KW = dict(kernel='poly', gamma=0.1, C=1.0)  # Best parameters specified
# ----------------------------

print("[*] Loading data ...")
X_sparse = load_npz("drebin_res_features.npz")
labels_df = pd.read_csv("labels.csv")
assert "family" in labels_df.columns, "labels.csv must contain column 'family'"

# 1. Global filtering: family samples ≥2
counts_all = labels_df["family"].value_counts()
valid_families = counts_all[counts_all >= 2].index.tolist()
n_filtered = len(counts_all) - len(valid_families)
print(f"[*] Families total: {len(counts_all)}; Filtering out {n_filtered} families with <2 samples (global).")
if len(valid_families) == 0:
    raise RuntimeError("No family has >=2 samples after global filtering.")

mask = labels_df["family"].isin(valid_families).values
indices_keep = np.where(mask)[0]
X_sparse = X_sparse[indices_keep]
labels_df = labels_df.iloc[indices_keep].reset_index(drop=True)

le = LabelEncoder()
y_all = le.fit_transform(labels_df["family"].values)
n_samples, n_features = X_sparse.shape
n_classes = len(le.classes_)
print(f"Loaded (after filter): samples={n_samples}, features={n_features}, classes={n_classes}")

# 2. Use RandomForest to assess importance (run once, fixed seed)
print("[*] Fitting RandomForest to get feature importance ...")
rf = RandomForestClassifier(n_estimators=200, random_state=RF_RANDOM_STATE, n_jobs=-1)
rf.fit(X_sparse, y_all)
imp = rf.feature_importances_
top_indices = np.argsort(imp)[-TOPK:][::-1]   # Descending order
print(f"[*] Selected top {TOPK} features by RF importance.")

# 3. Extract top 1000 dense matrix (directly use for later splitting)
X_topk = X_sparse[:, top_indices].toarray()

# ---------- Evaluation function ----------
def evaluate_once(train_idx, test_idx, random_state):
    # Ensure that each class has ≥2 samples in the training set (same as the original script)
    train_counts = Counter(y_all[train_idx])
    ok_classes = [c for c, cnt in train_counts.items() if cnt >= 2]
    train_idx = np.array([i for i in train_idx if y_all[i] in ok_classes], dtype=int)

    X_tr, X_te = X_topk[train_idx], X_topk[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]

    print(f"  Training SVM on {X_tr.shape[0]} samples ...")
    svm = SVC(**SVM_KW)
    svm.fit(X_tr, y_tr)

    y_pred = svm.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_te, y_pred, average="macro", zero_division=0)

    # Top-20 error rate families
    error_stats = []
    for c in np.unique(y_te):
        mask = (y_te == c)
        n_total = mask.sum()
        n_wrong = (y_pred[mask] != y_te[mask]).sum()
        if n_total > 0:
            error_rate = n_wrong / n_total
            cname = le.inverse_transform([c])[0]
            error_stats.append((cname, n_total, error_rate))
    top20 = sorted(error_stats, key=lambda x: x[2], reverse=True)[:20]
    print("\n  [Per-class error rate TOP-20]")
    print(f"  {'Class':25s} | {'#Samples':>8s} | {'Error rate':>10s}")
    print("  " + "-"*50)
    for cname, n_total, err in top20:
        print(f"  {cname:25s} | {n_total:8d} | {err:10.2%}")
    return acc, f1, rec

# ---------- 5 Repeats ----------
all_acc, all_f1, all_rec = [], [], []
for rep in range(1, REPEATS + 1):
    seed = RANDOM_SEED + rep * 100
    print(f"\n===== Repeat {rep}/{REPEATS} (seed={seed}) =====")
    ss = StratifiedShuffleSplit(n_splits=1, train_size=GLOBAL_TRAIN_RATIO, random_state=seed)
    train_idx, test_idx = next(ss.split(np.arange(n_samples), y_all))
    acc, f1, rec = evaluate_once(train_idx, test_idx, seed)
    print(f"Repeat {rep} -> Acc: {acc:.4f}, Macro-F1: {f1:.4f}, Macro-Recall: {rec:.4f}")
    all_acc.append(acc); all_f1.append(f1); all_rec.append(rec)

# ---------- Final Summary ----------
print("\n===== Summary over repeats =====")
print(f"Acc mean/std: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
print(f"Macro-F1 mean/std: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
print(f"Macro-Recall mean/std: {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
