# -*- coding: utf-8 -*-
"""Bài 5: so sánh Random Forest và SVC trước, sau cân bằng lớp."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Đọc và chia dữ liệu
df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Tạo mô hình thường và mô hình có cân bằng lớp
mo_hinh = {
    "Random Forest thuong": RandomForestClassifier(200, random_state=0),
    "Random Forest balanced": RandomForestClassifier(
        200, random_state=0, class_weight="balanced"
    ),
    "SVC thuong": make_pipeline(StandardScaler(), SVC(probability=True)),
    "SVC balanced": make_pipeline(
        StandardScaler(), SVC(probability=True, class_weight="balanced")
    )
}

# Huấn luyện và in các chỉ số
print("Mo hinh                     Precision  Recall     F1")
for ten, mh in mo_hinh.items():
    du = mh.fit(X_tr, y_tr).predict(X_te)
    p = precision_score(y_te, du, zero_division=0)
    r = recall_score(y_te, du, zero_division=0)
    f = f1_score(y_te, du, zero_division=0)
    print(f"{ten:27s} {p:.4f}     {r:.4f}   {f:.4f}")
