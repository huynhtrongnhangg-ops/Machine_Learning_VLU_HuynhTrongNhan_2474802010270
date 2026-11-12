# -*- coding: utf-8 -*-
"""Bài 6: so sánh thời gian học của SVC và LinearSVC."""

import time
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.svm import SVC, LinearSVC

# Các kích thước dữ liệu theo yêu cầu
kich_thuoc = [2000, 5000, 10000, 20000]
ket_qua = []

for n in kich_thuoc:
    # Sinh dữ liệu với 20 đặc trưng
    X, y = make_classification(
        n_samples=n,
        n_features=20,
        random_state=0
    )

    # Đo thời gian huấn luyện SVC
    svm = SVC()
    bat_dau = time.perf_counter()
    svm.fit(X, y)
    thoi_gian_svc = time.perf_counter() - bat_dau

    # Đo thời gian huấn luyện LinearSVC
    linear_svm = LinearSVC(random_state=0, max_iter=10000)
    bat_dau = time.perf_counter()
    linear_svm.fit(X, y)
    thoi_gian_linear = time.perf_counter() - bat_dau

    ket_qua.append({
        "So mau": n,
        "SVC (giay)": thoi_gian_svc,
        "LinearSVC (giay)": thoi_gian_linear
    })

# In bảng thời gian
bang = pd.DataFrame(ket_qua)
print("Bang thoi gian huan luyen")
print(bang.to_string(index=False, formatters={
    "SVC (giay)": lambda x: f"{x:.4f}",
    "LinearSVC (giay)": lambda x: f"{x:.4f}"
}))

# Tính tỷ lệ tăng thời gian giữa các kích thước liên tiếp
print()
print("Ty le tang thoi gian khi tang kich thuoc du lieu")
for i in range(1, len(ket_qua)):
    truoc = ket_qua[i - 1]
    sau = ket_qua[i]

    ty_le_svc = sau["SVC (giay)"] / truoc["SVC (giay)"]
    ty_le_linear = (
        sau["LinearSVC (giay)"] / truoc["LinearSVC (giay)"]
    )

    print(
        f"{truoc['So mau']} -> {sau['So mau']}: "
        f"SVC tang {ty_le_svc:.2f} lan, "
        f"LinearSVC tang {ty_le_linear:.2f} lan"
    )
