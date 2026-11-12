# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/hai_vang_trang.csv")
X, y = df[["x1", "x2"]], df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=17, stratify=y)
ket_qua = {}
print("Ham kich hoat   Tap hoc   Tap kiem tra")
for ten in ["logistic", "tanh", "relu"]:
    mang = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,100), activation=ten, alpha=1., solver="lbfgs", max_iter=20000, random_state=0))
    mang.fit(X_tr, y_tr)
    a = accuracy_score(y_tr, mang.predict(X_tr))
    b = accuracy_score(y_te, mang.predict(X_te))
    ket_qua[ten] = b
    print(f"{ten:14s} {a:.4f}    {b:.4f}")
print("Tot nhat tren tap kiem tra:", max(ket_qua, key=ket_qua.get))
