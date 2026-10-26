
# -*- coding: utf-8 -*-

"""
Bước 2: Đo hậu quả của việc quên chuẩn hóa trên tập kiểm tra.
"""

import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Đọc dữ liệu sinh viên
df = pd.read_csv("data/sinh_vien_mo_rong.csv")

# Lấy nhãn kết quả qua môn
y = df["qua_mon"]

# Lấy ba cột dữ liệu
X3 = df[["gio_on", "diem_giua_ky", "chi_tieu"]]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_tr, X_te, y_tr, y_te = train_test_split(
    X3,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)


# Hàm huấn luyện mô hình và tính độ chính xác
def cham(mo_hinh, cot):
    mo_hinh.fit(X_tr[cot], y_tr)
    return accuracy_score(
        y_te,
        mo_hinh.predict(X_te[cot])
    )


# Hàm tạo mô hình KNN có chuẩn hóa dữ liệu
def knn_chuan():
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=7)
    )


# Danh sách các cột dữ liệu
ba_cot = ["gio_on", "diem_giua_ky", "chi_tieu"]
hai_cot = ["gio_on", "diem_giua_ky"]


# Trường hợp A: Ba cột, không chuẩn hóa
a = cham(
    KNeighborsClassifier(n_neighbors=7),
    ba_cot
)

# Trường hợp B: Ba cột, có chuẩn hóa
b = cham(
    knn_chuan(),
    ba_cot
)

# Trường hợp C: Bỏ cột chi_tieu, có chuẩn hóa
c = cham(
    knn_chuan(),
    hai_cot
)


# In kết quả so sánh
print(f"Ty le qua mon trong tap kiem tra: {y_te.mean():.4f}")
print(f"Ba cot, khong chuan hoa: {a:.4f}")
print(f"Ba cot, co chuan hoa: {b:.4f}")
print(f"Bo cot chi_tieu, co chuan hoa: {c:.4f}")
