# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("data/xor.csv")
X, y = df[["x1", "x2"]], df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=17, stratify=y)
lr = LogisticRegression().fit(X_tr, y_tr)
print(f"Hoi quy logistic: {accuracy_score(y_te, lr.predict(X_te)):.4f}")
print("So no-ron an   so tham so   tap hoc   tap kiem tra")
for so in [1, 2, 4, 8]:
    mlp = MLPClassifier(hidden_layer_sizes=(so,), solver="lbfgs", max_iter=5000, random_state=0).fit(X_tr, y_tr)
    so_ts = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    print(f"{so:12d}   {so_ts:10d}   {accuracy_score(y_tr, mlp.predict(X_tr)):.4f}    {accuracy_score(y_te, mlp.predict(X_te)):.4f}")
