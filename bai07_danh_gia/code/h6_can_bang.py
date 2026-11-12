# -*- coding: utf-8 -*-
"""Bước 6: kiểm định chéo nhiều thước đo và cân bằng lớp."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu
df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]

# Thử chia không phân tầng để thấy số mẫu lớp hiếm thay đổi
print("Khong stratify, so ban bo hoc trong tap kiem tra:")
dem = []
for rs in range(10):
    _, _, _, y_te_thu = train_test_split(
        X, y, test_size=0.25, random_state=rs
    )
    dem.append(int(y_te_thu.sum()))
print(" ", dem)
print()

# Chia có phân tầng
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Kiểm định chéo với nhiều thước đo
lr = make_pipeline(StandardScaler(), LogisticRegression())
print("Kiem dinh cheo 5 phan tren tap hoc:")
for tm in ["accuracy", "precision", "recall", "f1"]:
    d = cross_val_score(lr, X_tr, y_tr, cv=5, scoring=tm)
    print(f" {tm:10s} {d.mean():.4f}")
print()

# So sánh mô hình bình thường và mô hình cân bằng lớp
cb = make_pipeline(
    StandardScaler(), LogisticRegression(class_weight="balanced")
)
print("Tap kiem tra   precision recall     F1")
for ten, mh in [("Binh thuong", lr), ("Can bang lop", cb)]:
    du = mh.fit(X_tr, y_tr).predict(X_te)
    print(
        f" {ten:12s} {precision_score(y_te, du, zero_division=0):.4f}"
        f"    {recall_score(y_te, du, zero_division=0):.4f}"
        f" {f1_score(y_te, du, zero_division=0):.4f}"
    )
