# -*- coding: utf-8 -*-
"""Bài 4: tự vẽ ROC của logistic và random forest, lưu roc.png."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Đọc và chia dữ liệu
df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Khai báo hai mô hình
mo_hinh = {
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "Random Forest": RandomForestClassifier(200, random_state=0)
}

# Vẽ hai đường ROC và ghi AUC vào chú thích
plt.figure(figsize=(8, 6))
ket_qua = {}

for ten, mh in mo_hinh.items():
    mh.fit(X_tr, y_tr)
    xs = mh.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, xs)
    auc = roc_auc_score(y_te, xs)
    ket_qua[ten] = (fpr, tpr, auc)
    plt.plot(fpr, tpr, label=f"{ten}, AUC = {auc:.4f}")

# Đường chéo biểu diễn khả năng đoán ngẫu nhiên
plt.plot([0, 1], [0, 1], linestyle="--", label="Doan ngau nhien")
plt.xlabel("Ty le bao dong nham (FPR)")
plt.ylabel("Recall (TPR)")
plt.title("So sanh duong ROC")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc.png", dpi=150)
plt.show()

# In AUC và gợi ý so sánh hai đường
print("Da luu do thi: roc.png")
print("AUC Logistic:", f"{ket_qua['Logistic'][2]:.4f}")
print("AUC Random Forest:", f"{ket_qua['Random Forest'][2]:.4f}")
print("Duong co AUC cao hon co xu huong tot hon tren toan bo cac nguong.")
print("Quan sat hinh roc.png de nhan xet cac doan hai duong cat nhau.")
