
# -*- coding: utf-8 -*-
"""Bài tập 2: tự tìm câu hỏi gốc bằng chỉ số Gini."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc dữ liệu khách vay
df = pd.read_csv("data/khach_vay.csv")

X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

# Chia dữ liệu, lấy 450 khách làm tập học
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def gini(nhan):
    """Tính chỉ số Gini của một nhóm dữ liệu."""
    if len(nhan) == 0:
        return 0.0

    ty_le = np.bincount(nhan, minlength=2) / len(nhan)

    return 1 - np.sum(ty_le ** 2)


def gini_sau_chia(y, dieu_kien):
    """Tính Gini trung bình có trọng số sau khi chia."""
    nhan_trai = y[dieu_kien]
    nhan_phai = y[~dieu_kien]

    tong = len(y)

    return (
        len(nhan_trai) / tong * gini(nhan_trai)
        + len(nhan_phai) / tong * gini(nhan_phai)
    )


def tim_cau_hoi(X, y):
    """Tìm cột và ngưỡng làm Gini giảm nhiều nhất."""
    gini_goc = gini(y)

    giam_gini_tot_nhat = -1
    cot_tot_nhat = None
    nguong_tot_nhat = None

    for ten_cot in X.columns:
        gia_tri = np.sort(X[ten_cot].unique())

        # Thử mọi ngưỡng nằm giữa hai giá trị liên tiếp
        for a, b in zip(gia_tri[:-1], gia_tri[1:]):
            nguong = (a + b) / 2

            dieu_kien = X[ten_cot].to_numpy() <= nguong

            gini_moi = gini_sau_chia(y, dieu_kien)

            giam_gini = gini_goc - gini_moi

            if giam_gini > giam_gini_tot_nhat:
                giam_gini_tot_nhat = giam_gini
                cot_tot_nhat = ten_cot
                nguong_tot_nhat = nguong

    return cot_tot_nhat, nguong_tot_nhat, giam_gini_tot_nhat


# Chạy hàm trên tập học gồm 450 khách
cot, nguong, giam_gini = tim_cau_hoi(
    X_tr,
    y_tr.to_numpy()
)

print("Ket qua tim cau hoi goc")
print(f"So khach tap hoc: {len(X_tr)}")
print(f"Cot tot nhat: {cot}")
print(f"Nguong tot nhat: {nguong:.2f}")
print(f"Gini giam: {giam_gini:.4f}")


# Nhận xét:
# Hàm đã thử các cột và ngưỡng để tìm câu hỏi chia dữ liệu tốt nhất.
# Kết quả cho thấy thu_nhap <= 15.05 là câu hỏi gốc tối ưu.
# Sau khi chia, chỉ số Gini giảm 0.1221 so với nút gốc.
# Điều này cho thấy câu hỏi này giúp phân chia nhóm khách vay
# có kết quả trả đúng hạn và trễ hạn rõ ràng hơn.
# Kết quả trùng với câu hỏi gốc của cây quyết định trong bài.
