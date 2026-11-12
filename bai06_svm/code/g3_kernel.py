# -*- coding: utf-8 -*-
"""Bước 3: nâng chiều bằng tay và thử các kernel."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Phần 1: dữ liệu một chiều, lớp 1 nằm giữa, lớp 0 ở hai đầu
x = np.array([
    -4.0, -3.4, -2.8, -2.2, -1.2, -0.7, -0.2,
    0.3, 0.8, 1.3, 2.3, 2.9, 3.5, 4.0
])
y = (np.abs(x) < 1.8).astype(int)

# Thử các ngưỡng trên trục x
do_chinh_x = max(
    ((x <= nguong).astype(int) == y).mean()
    for nguong in x
)
print(f"Mot nguong tren truc x, tot nhat dung {do_chinh_x:.4f}")

# Thêm cột x bình phương rồi dùng SVM tuyến tính
X_moi = np.c_[x, x ** 2]
svm_tuyen_tinh = SVC(kernel="linear", C=1000).fit(X_moi, y)
print(
    f"Them cot x^2, SVM tuyen tinh dung "
    f"{svm_tuyen_tinh.score(X_moi, y):.4f}"
)
print()

# Phần 2: dữ liệu hai vầng trăng
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# So sánh các mô hình
mo_hinh = {
    "SVM kernel tuyen tinh": SVC(kernel="linear"),
    "SVM kernel da thuc bac 3": SVC(kernel="poly", degree=3),
    "SVM kernel RBF": SVC(kernel="rbf"),
    "Hoi quy logistic": LogisticRegression(),
    "KNN, k = 13": KNeighborsClassifier(n_neighbors=13)
}

print("Hai vang trang, accuracy tren tap kiem tra")
for ten, mo_hinh_hien_tai in mo_hinh.items():
    mo_hinh_hien_tai.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, mo_hinh_hien_tai.predict(X_te))
    print(f" {ten:26s} {acc:.4f}")

# In số vectơ hỗ trợ của SVM RBF
svm_rbf = mo_hinh["SVM kernel RBF"]
print(
    "So vecto ho tro cua kernel RBF:",
    len(svm_rbf.support_),
    "tren",
    len(X_tr),
    "diem hoc"
)
