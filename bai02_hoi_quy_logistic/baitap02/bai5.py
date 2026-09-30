# -*- coding: utf-8 -*-
"""Bài tập 5: Dò ngưỡng tốt nhất theo F1."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)
p = mo_hinh.predict_proba(X_test)[:, 1]
nguong_tot_nhat = 0
f1_tot_nhat = 0
print("Nguong    F1")
for nguong in np.arange(0.05, 1.0, 0.05):
    y_pred = (p >= nguong).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"{nguong:.2f}     {f1:.4f}")
    if f1 > f1_tot_nhat:
        f1_tot_nhat = f1
        nguong_tot_nhat = nguong
print()
print(f"Nguong co F1 cao nhat: {nguong_tot_nhat:.2f}")
print(f"F1 cao nhat: {f1_tot_nhat:.4f}")
print()
if nguong_tot_nhat == 0.5:
    print("Nhan xet: Nguong tot nhat theo F1 bang 0.5.")
else:
    print("Nhan xet: Nguong tot nhat theo F1 khong bang 0.5.")