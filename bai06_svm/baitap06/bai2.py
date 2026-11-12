# -*- coding: utf-8 -*-
"""Bài 2: thêm từng điểm mới và quan sát đường ranh giới."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Tạo lại dữ liệu gốc
X_goc, y_goc = make_blobs(
    n_samples=24,
    centers=[[2, 2], [6, 6]],
    cluster_std=0.9,
    random_state=0
)
y_goc = 1 - y_goc

# Hàm huấn luyện và in đường ranh giới
def huan_luyen_va_in(X, y, ten):
    svm = SVC(kernel="linear", C=1000)
    svm.fit(X, y)
    w = svm.coef_[0]
    b = svm.intercept_[0]

    print(ten)
    print(f" w1 = {w[0]:.3f}, w2 = {w[1]:.3f}, b = {b:.3f}")
    print(" So vecto ho tro:", len(svm.support_))
    print(" Chi so vecto ho tro:", svm.support_)
    print()
    return svm

# Đường ranh giới ban đầu
svm_goc = huan_luyen_va_in(X_goc, y_goc, "Du lieu goc")

# Trường hợp 1: thêm điểm lớp 1 tại (-1, -1)
diem_a = np.array([[-1.0, -1.0]])
X_a = np.vstack([X_goc, diem_a])
y_a = np.append(y_goc, 1)
svm_a = huan_luyen_va_in(X_a, y_a, "Them diem lop 1 tai (-1, -1)")

# Trường hợp 2: thay bằng điểm lớp 1 tại (4.5, 4)
diem_b = np.array([[4.5, 4.0]])
X_b = np.vstack([X_goc, diem_b])
y_b = np.append(y_goc, 1)
svm_b = huan_luyen_va_in(X_b, y_b, "Them diem lop 1 tai (4.5, 4)")

# So sánh hệ số để nhận biết đường ranh giới có thay đổi hay không
print("So sanh voi du lieu goc")
for ten, svm_moi in [
    ("Diem (-1, -1)", svm_a),
    ("Diem (4.5, 4)", svm_b)
]:
    thay_doi = not (
        np.allclose(svm_goc.coef_[0], svm_moi.coef_[0], atol=1e-3)
        and np.allclose(svm_goc.intercept_, svm_moi.intercept_, atol=1e-3)
    )
    print(f" {ten}: {'co thay doi' if thay_doi else 'gan nhu khong thay doi'}")
