# -*- coding: utf-8 -*-
"""Bài 3: chọn ngưỡng lớn nhất để recall kiểm định chéo đạt ít nhất 0.8."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Đọc và chia dữ liệu
df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Xác suất dự đoán chéo trên tập học, không dùng tập kiểm tra để chọn ngưỡng
lr = make_pipeline(StandardScaler(), LogisticRegression())
xs_hoc = cross_val_predict(
    lr, X_tr, y_tr, cv=5, method="predict_proba"
)[:, 1]

# Tìm ngưỡng lớn nhất có recall từ 0.8 trở lên
cac_nguong = np.unique(np.r_[0.0, xs_hoc])
cac_nguong = np.sort(cac_nguong)[::-1]
t_tot = None
recall_cv = 0

for t in cac_nguong:
    du_cv = (xs_hoc >= t).astype(int)
    r = recall_score(y_tr, du_cv, zero_division=0)
    if r >= 0.8:
        t_tot = float(t)
        recall_cv = r
        break

print(f"Nguong duoc chon tren tap hoc: {t_tot:.6f}")
print(f"Recall kiem dinh cheo tren tap hoc: {recall_cv:.4f}")

# Huấn luyện trên toàn bộ tập học và chỉ áp ngưỡng lên tập kiểm tra một lần
lr.fit(X_tr, y_tr)
xs_te = lr.predict_proba(X_te)[:, 1]
du_te = (xs_te >= t_tot).astype(int)

print("Ket qua tren tap kiem tra:")
print(f" So ban duoc bao: {int(du_te.sum())}")
print(f" Precision: {precision_score(y_te, du_te, zero_division=0):.4f}")
print(f" Recall:    {recall_score(y_te, du_te, zero_division=0):.4f}")
print(f" F1:        {f1_score(y_te, du_te, zero_division=0):.4f}")
print("Recall tren tap kiem tra co dat 0.8 hay khong:",
      recall_score(y_te, du_te, zero_division=0) >= 0.8)
