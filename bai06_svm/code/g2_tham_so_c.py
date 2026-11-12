# -*- coding: utf-8 -*-
"""Bước 2: lề mềm, tham số C và việc chuẩn hóa dữ liệu."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Đọc dữ liệu sinh viên
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]

# Chia tập học và tập kiểm tra
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("Tap hoc co", len(X_tr), "ban")
print("    C    so vecto HT   do rong le    tap hoc   tap kiem tra")

# Thử nhiều giá trị C
for C in [0.01, 0.1, 1, 10, 100]:
    mo_hinh = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=C)
    )
    mo_hinh.fit(X_tr, y_tr)

    # Bước cuối của pipeline là mô hình SVM
    svm = mo_hinh[-1]
    do_rong_le = 2 / np.linalg.norm(svm.coef_[0])
    acc_hoc = accuracy_score(y_tr, mo_hinh.predict(X_tr))
    acc_kiem_tra = accuracy_score(y_te, mo_hinh.predict(X_te))

    print(
        f" {C:6}    {len(svm.support_):6d}        {do_rong_le:6.2f}"
        f"      {acc_hoc:.4f}     {acc_kiem_tra:.4f}"
    )
print()

# Đọc bộ dữ liệu mở rộng có thêm cột chi tiêu
df_mo_rong = pd.read_csv("data/sinh_vien_mo_rong.csv")
X_mo_rong = df_mo_rong[["gio_on", "diem_giua_ky", "chi_tieu"]]
y_mo_rong = df_mo_rong["qua_mon"]

X3_tr, X3_te, y3_tr, y3_te = train_test_split(
    X_mo_rong,
    y_mo_rong,
    test_size=0.25,
    random_state=17,
    stratify=y_mo_rong
)

# So sánh SVM không chuẩn hóa và có chuẩn hóa
svm_tho = SVC().fit(X3_tr, y3_tr)
svm_chuan_hoa = make_pipeline(
    StandardScaler(),
    SVC()
).fit(X3_tr, y3_tr)

print("Ba cot, kernel RBF mac dinh")
print(f" Doan bua ca lop qua : {y3_te.mean():.4f}")
print(
    f" Khong chuan hoa      : "
    f"{accuracy_score(y3_te, svm_tho.predict(X3_te)):.4f}"
)
print(
    f" Co chuan hoa         : "
    f"{accuracy_score(y3_te, svm_chuan_hoa.predict(X3_te)):.4f}"
)
