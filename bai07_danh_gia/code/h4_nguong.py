# -*- coding: utf-8 -*-
"""Bước 4: đổi ngưỡng và chọn ngưỡng đúng cách trên tập học."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
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

# Huấn luyện logistic và lấy xác suất trên tập kiểm tra
lr = make_pipeline(StandardScaler(), LogisticRegression())
lr.fit(X_tr, y_tr)
xs = lr.predict_proba(X_te)[:, 1]

# Quan sát ảnh hưởng của các ngưỡng trên tập kiểm tra
print("Tap kiem tra, doi nguong")
print("nguong so bao    TP FP FN     precision recall     F1")
for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
    du = (xs >= t).astype(int)
    tp = int(((du == 1) & (y_te == 1)).sum())
    fp = int(((du == 1) & (y_te == 0)).sum())
    fn = int(((du == 0) & (y_te == 1)).sum())
    print(
        f" {t:.1f}     {du.sum():4d}   {tp:3d} {fp:3d} {fn:3d}"
        f"    {precision_score(y_te, du, zero_division=0):.4f}"
        f"   {recall_score(y_te, du, zero_division=0):.4f}"
        f" {f1_score(y_te, du, zero_division=0):.4f}"
    )
print()

# Tạo xác suất dự đoán chéo chỉ từ tập học
xs_hoc = cross_val_predict(
    lr, X_tr, y_tr, cv=5, method="predict_proba"
)[:, 1]

# Chọn ngưỡng cho F1 cao nhất trên tập học
cac_nguong = np.round(np.arange(0.05, 0.96, 0.05), 2)
f1_hoc = [f1_score(y_tr, (xs_hoc >= t).astype(int)) for t in cac_nguong]
t_tot = cac_nguong[int(np.argmax(f1_hoc))]
print(
    f"Nguong cho F1 cao nhat tren tap hoc: {t_tot}"
    f" (F1 kiem dinh cheo {max(f1_hoc):.4f})"
)

# Chỉ đánh giá tập kiểm tra sau khi đã chọn ngưỡng
du = (xs >= t_tot).astype(int)
print(
    f"Tap kiem tra voi nguong {t_tot}:"
    f" precision {precision_score(y_te, du, zero_division=0):.4f},"
    f" recall {recall_score(y_te, du, zero_division=0):.4f},"
    f" F1 {f1_score(y_te, du, zero_division=0):.4f}"
)
