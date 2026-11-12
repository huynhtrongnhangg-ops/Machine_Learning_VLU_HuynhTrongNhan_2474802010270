# -*- coding: utf-8 -*-
"""Bài 6: đo tác động của một dự đoán sai lớn lên MAE, RMSE và R2."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Đọc dữ liệu giá nhà của bài 1
df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

# Chia dữ liệu giống bài 1
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Huấn luyện hồi quy tuyến tính và dự đoán
lr = LinearRegression()
lr.fit(X_tr, y_tr)
du_goc = lr.predict(X_te)

# Hàm tính ba thước đo
def tinh_chi_so(y_that, y_doan):
    mae = mean_absolute_error(y_that, y_doan)
    rmse = np.sqrt(mean_squared_error(y_that, y_doan))
    r2 = r2_score(y_that, y_doan)
    return mae, rmse, r2

# Đo kết quả ban đầu
mae_1, rmse_1, r2_1 = tinh_chi_so(y_te, du_goc)
print("Truoc khi them sai so:")
print(f" MAE  = {mae_1:.4f} ty dong")
print(f" RMSE = {rmse_1:.4f} ty dong")
print(f" R2   = {r2_1:.4f}")

# Cộng thêm 5 tỷ vào dự đoán của đúng một căn
du_sau = du_goc.copy()
du_sau[0] += 5
print()
print("Can bi thay doi la mau dau tien trong tap kiem tra.")
print("Sai so duoc cong them 5 ty dong vao gia du doan.")

# Đo lại sau khi cố ý làm hỏng một dự đoán
mae_2, rmse_2, r2_2 = tinh_chi_so(y_te, du_sau)
print("Sau khi them sai so:")
print(f" MAE  = {mae_2:.4f} ty dong")
print(f" RMSE = {rmse_2:.4f} ty dong")
print(f" R2   = {r2_2:.4f}")

print()
print(f"Muc thay doi MAE  = {abs(mae_2 - mae_1):.4f}")
print(f"Muc thay doi RMSE = {abs(rmse_2 - rmse_1):.4f}")
print(f"Muc thay doi R2   = {abs(r2_2 - r2_1):.4f}")
print("RMSE binh phuong sai so truoc khi lay can, nen phat nang sai so lon.")
