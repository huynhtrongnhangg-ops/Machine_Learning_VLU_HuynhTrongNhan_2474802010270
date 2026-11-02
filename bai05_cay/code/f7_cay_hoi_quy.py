
# -*- coding: utf-8 -*-
"""Bước 7: dùng cây hồi quy để dự đoán giá nhà."""

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Đọc dữ liệu giá nhà
df = pd.read_csv("data/gia_nha.csv")

print("Kich thuoc bang:", df.shape)
print(df.head())
print()

# Chọn diện tích làm đặc trưng và giá nhà làm mục tiêu
X = df[["dien_tich"]]
y = df["gia"]

# Chia dữ liệu thành tập học và tập kiểm tra
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cây hồi quy giới hạn độ sâu 3
cay = DecisionTreeRegressor(max_depth=3, random_state=0)

cay.fit(X_tr, y_tr)

# Dự đoán giá nhà trên tập kiểm tra
du_doan = cay.predict(X_te)

mae = mean_absolute_error(y_te, du_doan)

print("Cay hoi quy")
print(f"Do sau: {cay.get_depth()}")
print(f"So la: {cay.get_n_leaves()}")
print(f"MAE tap kiem tra: {mae:.4f}")
print()

# Dự đoán giá nhà theo diện tích
dien_tich_moi = pd.DataFrame({
    "dien_tich": [60, 100, 150, 250]
})

gia_du_doan = cay.predict(dien_tich_moi)

print("Du doan gia nha")

for i in range(len(dien_tich_moi)):
    print(
        f"Dien tich {dien_tich_moi.iloc[i, 0]} m2: "
        f"{gia_du_doan[i]:.4f} ty"
    )
