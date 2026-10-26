
# -*- coding: utf-8 -*-

"""
Bước 4: Đặt KNN cạnh hai mô hình đã học trên dữ liệu có biên cong.
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# Đọc dữ liệu hai loại vàng và trắng
df = pd.read_csv("data/hai_vang_trang.csv")

# Lấy hai cột đặc trưng và nhãn
X = df[["x1", "x2"]]
y = df["nhan"]


# Chia dữ liệu thành tập học và tập kiểm tra
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)


# Khởi tạo ba mô hình cần so sánh
cac_mo_hinh = [
    ("Hoi quy logistic", LogisticRegression()),
    ("Naive Bayes", GaussianNB()),
    ("KNN, k = 13", KNeighborsClassifier(n_neighbors=13)),
]


# Huấn luyện và so sánh độ chính xác trên tập kiểm tra
print("Accuracy tren tap kiem tra, bo hai vang trang")

for ten, mh in cac_mo_hinh:
    mh.fit(X_tr, y_tr)

    acc = accuracy_score(
        y_te,
        mh.predict(X_te)
    )

    print(f" {ten}: {acc:.4f}")
