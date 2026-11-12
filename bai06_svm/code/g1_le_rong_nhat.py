# -*- coding: utf-8 -*-
"""Bước 1: tìm đường có lề rộng nhất và các vectơ hỗ trợ."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Tạo 24 điểm thuộc hai cụm tách biệt
X, y = make_blobs(
    n_samples=24,
    centers=[[2, 2], [6, 6]],
    cluster_std=0.9,
    random_state=0
)

# Đảo nhãn để lớp 1 là cụm phía dưới bên trái
y = 1 - y

# Huấn luyện SVM tuyến tính với C lớn
svm = SVC(kernel="linear", C=1000)
svm.fit(X, y)

# Lấy hệ số của đường ranh giới
w = svm.coef_[0]
b = svm.intercept_[0]

print(f"w1 = {w[0]:.3f}, w2 = {w[1]:.3f}, b = {b:.3f}")

# Tính độ rộng lề
do_dai_w = np.sqrt(w[0] ** 2 + w[1] ** 2)
print(f"Do rong cua le = 2 / {do_dai_w:.4f} = {2 / do_dai_w:.4f}")
print()

# In các vectơ hỗ trợ
print("So vecto ho tro:", len(svm.support_))
for i in svm.support_:
    gia_tri_f = svm.decision_function(X[i:i + 1])[0]
    print(
        f" diem {i:2d}: x1 = {X[i, 0]:.2f}, x2 = {X[i, 1]:.2f},"
        f" lop {y[i]}, f(x) = {gia_tri_f:+.3f}"
    )
print()

# Kiểm tra các điểm không phải vectơ hỗ trợ
f_tat_ca = svm.decision_function(X)
khong_phai_sv = np.setdiff1d(np.arange(len(X)), svm.support_)
nho_nhat = np.abs(f_tat_ca[khong_phai_sv]).min()
print(f"|f(x)| nho nhat cua cac diem khac: {nho_nhat:.3f}")
print()

# Huấn luyện lại chỉ bằng các vectơ hỗ trợ
svm_chi_sv = SVC(kernel="linear", C=1000)
svm_chi_sv.fit(X[svm.support_], y[svm.support_])

w_moi = svm_chi_sv.coef_[0]
b_moi = svm_chi_sv.intercept_[0]

print("Hoc lai chi voi vecto ho tro:")
print(f" w1 = {w_moi[0]:.3f}, w2 = {w_moi[1]:.3f}, b = {b_moi:.3f}")
print()

# Dự đoán hai điểm mới
diem_moi = np.array([[3.0, 3.0], [5.0, 4.5]])
for diem, gia_tri_f, nhan in zip(
    diem_moi,
    svm.decision_function(diem_moi),
    svm.predict(diem_moi)
):
    print(
        f"Diem ({diem[0]}, {diem[1]}): "
        f"f(x) = {gia_tri_f:+.3f}, doan lop {nhan}"
    )
