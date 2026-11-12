# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Học mạng XOR giống mục 3
bang = pd.read_csv("data/xor.csv")
X, y = bang[["x1", "x2"]], bang["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=17, stratify=y)
mlp = MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=5000, random_state=0).fit(X_tr, y_tr)

# Tính bằng numpy: lớp ẩn ReLU, lớp ra sigmoid
nam_diem = X_te.iloc[:5].to_numpy()
lop_an = np.maximum(0, nam_diem @ mlp.coefs_[0] + mlp.intercepts_[0])
z_ra = lop_an @ mlp.coefs_[1] + mlp.intercepts_[1]
tu_tinh = expit(z_ra).ravel()
thu_vien = mlp.predict_proba(nam_diem)[:, 1]
print("Diem  Tu tinh     Thu vien    Sai lech")
for i, (a, b) in enumerate(zip(tu_tinh, thu_vien), 1):
    print(f"{i:4d}  {a:.10f}  {b:.10f}  {abs(a-b):.2e}")
print("Sai lech lon nhat:", np.max(np.abs(tu_tinh - thu_vien)))
print("Nhan xet: hai cach khop nhau vi deu dung cung trong so, ReLU va sigmoid.")
