
# -*- coding: utf-8 -*-
"""Bước 4: kiểm chứng cây không cần chuẩn hoá dữ liệu."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data/khach_vay.csv")

X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def cay_sau_5():
    return DecisionTreeClassifier(max_depth=5, random_state=0)


# Cây 1: dữ liệu gốc
goc = cay_sau_5().fit(X_tr, y_tr)
du_goc = goc.predict(X_te)


# Cây 2: dữ liệu đã chuẩn hoá, trung bình tính trên riêng tập học
sc = StandardScaler().fit(X_tr)

chuan = cay_sau_5().fit(sc.transform(X_tr), y_tr)

du_chuan = chuan.predict(sc.transform(X_te))


# Cây 3: nhân các cột với hệ số khác nhau
he_so = [100, 10, 1, 5]

X_tr_nhan = X_tr * he_so
X_te_nhan = X_te * he_so

nhan = cay_sau_5().fit(X_tr_nhan, y_tr)

du_nhan = nhan.predict(X_te_nhan)


# So sánh accuracy trên tập kiểm tra
print("So sanh ba cay sau 5")
print()

print(f"Cay du lieu goc       : {accuracy_score(y_te, du_goc):.4f}")
print(f"Cay du lieu chuan hoa  : {accuracy_score(y_te, du_chuan):.4f}")
print(f"Cay du lieu nhan he so : {accuracy_score(y_te, du_nhan):.4f}")

print()

# So sánh dự đoán của ba cây
print("So sanh du doan")

print(
    "Goc va chuan hoa:",
    "GIONG NHAU" if (du_goc == du_chuan).all() else "KHAC NHAU"
)

print(
    "Goc va nhan he so:",
    "GIONG NHAU" if (du_goc == du_nhan).all() else "KHAC NHAU"
)
