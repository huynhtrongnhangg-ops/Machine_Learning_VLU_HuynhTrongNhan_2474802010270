# -*- coding: utf-8 -*-
"""Bài tập 3: Dự đoán cho một bạn cụ thể."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]
mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)
w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def du_doan(gio):
    z = w * gio + b
    xac_suat = sigmoid(z)
    nhan = 1 if xac_suat >= 0.5 else 0
    print(f"So gio on: {gio:.2f}")
    print(f"z = {z:.4f}")
    print(f"Xac suat qua mon = {xac_suat:.4f}")
    print(f"Nhãn du doan = {nhan}")
    print()
for gio in [3, 8, 12.89, 18, 26]:
    du_doan(gio)
print("Nhan xet:")
print("Tai 12.89 gio, z gan bang 0 nen xac suat qua mon gan bang 0.5.")