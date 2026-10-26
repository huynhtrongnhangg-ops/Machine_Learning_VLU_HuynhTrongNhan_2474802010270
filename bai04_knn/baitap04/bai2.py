# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
#Chia dữ liệu giống bài e3
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
# So sánh khoảng cách Euclid và Manhattan
for ten, metric in [
    ("Euclid", "euclidean"),
    ("Manhattan", "manhattan")
]:
    diem = {}
    # Chọn k bằng kiểm định chéo trên tập học
    for k in range(1, 52, 2):
        mo_hinh = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric
        )
        diem[k] = cross_val_score(
            mo_hinh, X_tr, y_tr, cv=5
        ).mean()D
    k_tot = max(diem, key=diem.get)
    # Đánh giá trên tập kiểm tra sau khi chọn k
    mo_hinh = KNeighborsClassifier(
        n_neighbors=k_tot,
        metric=metric
    )
    mo_hinh.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, mo_hinh.predict(X_te))
    print(f"Khoang cach: {ten}")
    print(f"k tot nhat: {k_tot}")
    print(f"Accuracy tap kiem tra: {acc:.4f}")
    print()
