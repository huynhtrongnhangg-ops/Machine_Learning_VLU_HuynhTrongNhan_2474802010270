# -*- coding: utf-8 -*-
"""Bài 1: tính tay giá trị f(x) và khoảng cách tới ranh giới."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Tạo lại đúng 24 điểm như mục 2 của tài liệu
X, y = make_blobs(
    n_samples=24,
    centers=[[2, 2], [6, 6]],
    cluster_std=0.9,
    random_state=0
)
y = 1 - y

# Huấn luyện SVM tuyến tính
svm = SVC(kernel="linear", C=1000)
svm.fit(X, y)

# Lấy w1, w2, b từ mô hình mục 2
w1, w2 = svm.coef_[0]
b = svm.intercept_[0]
do_dai_w = np.sqrt(w1 ** 2 + w2 ** 2)

print("He so duong ranh gioi")
print(f"w1 = {w1:.3f}, w2 = {w2:.3f}, b = {b:.3f}")
print(f"Do dai w = {do_dai_w:.4f}")
print()

# Tự tính f(x) bằng phép nhân và cộng
diem_moi = [(3.0, 3.0), (5.0, 4.5)]
for x1, x2 in diem_moi:
    gia_tri_f = w1 * x1 + w2 * x2 + b
    khoang_cach = abs(gia_tri_f) / do_dai_w
    print(f"Diem ({x1}, {x2})")
    print(f" f(x) = {w1:.3f}*{x1} + {w2:.3f}*{x2} + {b:.3f} = {gia_tri_f:.4f}")
    print(f" Khoang cach toi ranh gioi = {abs(gia_tri_f):.4f}/{do_dai_w:.4f} = {khoang_cach:.4f}")
print()

# Tính khoảng cách của ba vectơ hỗ trợ
print("Khoang cach cua ba vecto ho tro")
for i in svm.support_:
    x1, x2 = X[i]
    gia_tri_f = w1 * x1 + w2 * x2 + b
    khoang_cach = abs(gia_tri_f) / do_dai_w
    print(
        f" Diem {i:2d}: f(x) = {gia_tri_f:+.4f}, "
        f"khoang cach = {khoang_cach:.4f}"
    )

# So sánh với một nửa độ rộng lề
nua_le = 1 / do_dai_w
do_rong_le = 2 / do_dai_w
print()
print(f"Mot nua do rong le = {nua_le:.4f}")
print(f"Do rong le = {do_rong_le:.4f}")
print(
    "Nhan xet: cac vecto ho tro nam tren hai mep le f(x) = +1 "
    "hoac f(x) = -1, nen khoang cach den ranh gioi bang 1/||w||."
)
