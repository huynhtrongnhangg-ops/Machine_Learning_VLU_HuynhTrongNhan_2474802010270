# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
#Chia dữ liệu giống bài e3
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
# So sánh KNN thường và KNN có trọng số khoảng cách
print("So sanh KNN thuong va KNN co trong so")
print(" k    KNN thuong    weights=distance")
for k in [1, 13, 101]:
    #KNN thông thường
    knn_thuong = KNeighborsClassifier(
        n_neighbors=k
    )
    knn_thuong.fit(X_tr, y_tr)

    acc_thuong = accuracy_score(
        y_te, knn_thuong.predict(X_te)
    )
    #KNN có trọng số khoảng cách
    knn_trong_so = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance"
    )
    knn_trong_so.fit(X_tr, y_tr)

    acc_trong_so = accuracy_score(
        y_te, knn_trong_so.predict(X_te)
    )
    print(f" {k:3d}    {acc_thuong:.4f}          {acc_trong_so:.4f}")
