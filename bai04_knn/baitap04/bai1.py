# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("data/hai_vang_trang.csv")
#Lấy hai cột đặc trưng và nhãn
X = df[["x1", "x2"]].to_numpy(dtype=float)
y = df["nhan"].to_numpy()
#Chia dữ liệu thành tập học và tập kiểm tra
X_hoc, X_kiem_tra, y_hoc, y_kiem_tra = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)
#Hàm KNN tự viết
def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    #Tính khoảng cách Euclid từ mẫu mới đến các mẫu học
    khoang_cach = np.sqrt(((X_hoc - x_moi) ** 2).sum(axis=1))
    #Lấy chỉ số của k láng giềng gần nhất
    chi_so = np.argsort(khoang_cach)[:k]
    #Lấy nhãn của k láng giềng
    nhan_lang_gieng = y_hoc[chi_so]
    #Đếm số phiếu của từng nhãn
    nhan, so_phieu = np.unique(
        nhan_lang_gieng,
        return_counts=True
    )
    #Trả về nhãn có nhiều phiếu nhất
    return nhan[np.argmax(so_phieu)]
#Dự đoán toàn bộ tập kiểm tra bằng KNN tự viết
y_du_doan = np.array([
    knn_du_doan(X_hoc, y_hoc, x, 13)
    for x in X_kiem_tra
])
#Tính accuracy của KNN tự viết
acc_tu_viet = accuracy_score(y_kiem_tra, y_du_doan)
#So sánh với KNeighborsClassifier của sklearn
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_hoc, y_hoc)
y_du_doan_sklearn = knn.predict(X_kiem_tra)
acc_sklearn = accuracy_score(y_kiem_tra, y_du_doan_sklearn)
#In kết quả
print(f"Accuracy KNN tu viet: {acc_tu_viet:.4f}")
print(f"Accuracy sklearn:     {acc_sklearn:.4f}")
print(
    "So sanh ket qua:",
    "GIONG NHAU" if np.array_equal(
        y_du_doan, y_du_doan_sklearn
    ) else "KHAC NHAU"
)
