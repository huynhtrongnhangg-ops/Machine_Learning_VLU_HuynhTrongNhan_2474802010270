
# -*- coding: utf-8 -*-
"""Bước 5: trồng rừng ngẫu nhiên và so với một cây."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data/khach_vay.csv")

X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Bốc mẫu có hoàn lại: rút 450 lần từ 450 khách
rng = np.random.default_rng(0)

n = len(X_tr)

chi_so = rng.integers(0, n, n)

khac = len(np.unique(chi_so))

print(f"Boc {n} lan, co {khac} khach khac nhau ({khac / n:.0%})")
print()

# Rừng ngẫu nhiên gồm 200 cây
rung = RandomForestClassifier(
    n_estimators=200,
    random_state=0
)

rung.fit(X_tr, y_tr)

print("Rung 200 cay")

print(f" tap hoc       : {accuracy_score(y_tr, rung.predict(X_tr)):.4f}")
print(f" tap kiem tra : {accuracy_score(y_te, rung.predict(X_te)):.4f}")

print()

# Một cây quyết định để so sánh
cay = DecisionTreeClassifier(random_state=0)

cay.fit(X_tr, y_tr)

print("Mot cay quyet dinh")

print(f" tap hoc       : {accuracy_score(y_tr, cay.predict(X_tr)):.4f}")
print(f" tap kiem tra : {accuracy_score(y_te, cay.predict(X_te)):.4f}")
