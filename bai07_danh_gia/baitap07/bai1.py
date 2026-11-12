# -*- coding: utf-8 -*-
"""Bài 1: tự viết hàm tính accuracy, precision, recall và F1 bằng numpy."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Hàm tự tính các chỉ số từ nhãn thật và nhãn dự đoán
def danh_gia(y_that, y_doan):
    y_that = np.asarray(y_that)
    y_doan = np.asarray(y_doan)

    tp = np.sum((y_that == 1) & (y_doan == 1))
    fp = np.sum((y_that == 0) & (y_doan == 1))
    fn = np.sum((y_that == 1) & (y_doan == 0))
    tn = np.sum((y_that == 0) & (y_doan == 0))

    tong = tp + fp + fn + tn
    accuracy = (tp + tn) / tong if tong else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return accuracy, precision, recall, f1

# Đọc dữ liệu và chia tập giống ví dụ trong tài liệu
df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Tạo hai mô hình
mo_hinh = {
    "Mo hinh luoi": DummyClassifier(strategy="most_frequent"),
    "Hoi quy logistic": make_pipeline(StandardScaler(), LogisticRegression())
}

# So sánh hàm tự viết với scikit-learn
for ten, mh in mo_hinh.items():
    du = mh.fit(X_tr, y_tr).predict(X_te)
    kq = danh_gia(y_te, du)

    print(ten)
    print(
        f" Tu viet: accuracy={kq[0]:.4f}, precision={kq[1]:.4f}, "
        f"recall={kq[2]:.4f}, F1={kq[3]:.4f}"
    )
    print(
        f" sklearn: accuracy={accuracy_score(y_te, du):.4f}, "
        f"precision={precision_score(y_te, du, zero_division=0):.4f}, "
        f"recall={recall_score(y_te, du, zero_division=0):.4f}, "
        f"F1={f1_score(y_te, du, zero_division=0):.4f}"
    )
    print()

print("Mo hinh luoi khong du doan mau duong nao, nen TP + FP = 0.")
print("Precision khong the chia truc tiep cho 0; ham xu ly bang cach tra ve 0.")
