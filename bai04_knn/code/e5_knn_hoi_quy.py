
# -*- coding: utf-8 -*-

"""
Bước 5: KNN cho bài toán hồi quy,
trên dữ liệu giá nhà của bài 1.
"""

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# Đọc dữ liệu giá nhà
df = pd.read_csv("data/gia_nha.csv")

# Lấy diện tích làm đặc trưng và giá nhà làm giá trị cần dự đoán
X = df[["dien_tich"]]
y = df["gia"]


# Chia dữ liệu giống bài 1 để so sánh công bằng
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Huấn luyện mô hình KNN hồi quy với 5 láng giềng
knn = KNeighborsRegressor(n_neighbors=5).fit(X_tr, y_tr)

# Huấn luyện mô hình hồi quy tuyến tính
duong = LinearRegression().fit(X_tr, y_tr)


# So sánh sai số MAE trên tập kiểm tra
print("MAE tren tap kiem tra (ty dong)")

mae_knn = mean_absolute_error(y_te, knn.predict(X_te))
mae_duong = mean_absolute_error(y_te, duong.predict(X_te))

print(f" KNN, k = 5 : {mae_knn:.4f}")
print(f" Duong thang : {mae_duong:.4f}")
print()


# Tạo dữ liệu của bốn căn nhà mới
can_moi = pd.DataFrame({
    "dien_tich": [60.0, 100.0, 150.0, 250.0]
})

# Dự đoán giá nhà bằng cả hai mô hình
print("Du doan cho bon can moi (ty dong)")
print(" dien tich   KNN   duong thang")

for dt, a, b in zip(
    can_moi["dien_tich"],
    knn.predict(can_moi),
    duong.predict(can_moi)
):
    print(f" {dt:6.0f} m2   {a:6.2f}   {b:6.2f}")

print()


# Tìm năm láng giềng gần nhất của căn nhà rộng 250 m2
_, chi_so = knn.kneighbors(
    pd.DataFrame({"dien_tich": [250.0]})
)

# Lấy diện tích của năm căn nhà láng giềng
lg = X_tr.iloc[chi_so[0]]["dien_tich"].tolist()

print(
    "Nam lang gieng cua can 250 m2:",
    ", ".join(f"{v:.1f}" for v in lg)
)
