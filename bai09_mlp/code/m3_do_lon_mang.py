# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/hai_vang_trang.csv")
X, y = df[["x1", "x2"]], df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=17, stratify=y)

def mang(lop_an, alpha):
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=lop_an, alpha=alpha, solver="lbfgs", max_iter=20000, random_state=0))

print("Ba mang                  tap hoc   tap kiem tra")
for ten, lop_an, alpha in [("(2,), alpha 1e-4", (2,), 1e-4), ("(100, 100), 1e-5", (100,100), 1e-5), ("(100, 100), 1.0", (100,100), 1.)]:
    m = mang(lop_an, alpha).fit(X_tr, y_tr)
    print(f"{ten:22s} {accuracy_score(y_tr, m.predict(X_tr)):.4f}    {accuracy_score(y_te, m.predict(X_te)):.4f}")
print("alpha      kiem dinh cheo")
diem = {}
for alpha in [1e-5, 1e-3, .1, 1., 3., 10.]:
    diem[alpha] = cross_val_score(mang((100,100), alpha), X_tr, y_tr, cv=5).mean()
    print(f"{alpha:<8}   {diem[alpha]:.4f}")
a_tot = max(diem, key=diem.get)
m = mang((100,100), a_tot).fit(X_tr, y_tr)
print(f"alpha tot nhat = {a_tot}, tap kiem tra {accuracy_score(y_te, m.predict(X_te)):.4f}")
