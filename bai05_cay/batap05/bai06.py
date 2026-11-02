
# -*- coding: utf-8 -*-
"""Bài tập 6: so sánh các mô hình dự đoán giá nhà."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Đọc dữ liệu giá nhà
df = pd.read_csv("data/gia_nha.csv")

X = df[["dien_tich"]]
y = df["gia"]

# Chia dữ liệu theo cách ở mục 8
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.25,
    random_state=17
)

# Mô hình 1: cây hồi quy sâu 3
cay = DecisionTreeRegressor(
    max_depth=3,
    random_state=0
)

cay.fit(X_tr, y_tr)

du_doan_cay = cay.predict(X_te)

mae_cay = mean_absolute_error(
    y_te,
    du_doan_cay
)

# Mô hình 2: hồi quy tuyến tính
duong_thang = LinearRegression()

duong_thang.fit(X_tr, y_tr)

du_doan_duong_thang = duong_thang.predict(X_te)

mae_duong_thang = mean_absolute_error(
    y_te,
    du_doan_duong_thang
)

# Mô hình 3: rừng ngẫu nhiên gồm 200 cây
rung = RandomForestRegressor(
    n_estimators=200,
    random_state=0
)

rung.fit(X_tr, y_tr)

du_doan_rung = rung.predict(X_te)

mae_rung = mean_absolute_error(
    y_te,
    du_doan_rung
)

# In kết quả MAE
print("So sanh MAE")
print(f"Cay hoi quy sau 3: {mae_cay:.4f}")
print(f"Duong thang: {mae_duong_thang:.4f}")
print(f"Rung ngau nhien: {mae_rung:.4f}")

# Dự đoán giá căn nhà 250 m2
nha_250 = pd.DataFrame({
    "dien_tich": [250]
})

gia_cay = cay.predict(nha_250)[0]
gia_duong_thang = duong_thang.predict(nha_250)[0]
gia_rung = rung.predict(nha_250)[0]

print()
print("Du doan gia nha 250 m2")
print(f"Cay hoi quy: {gia_cay:.4f}")
print(f"Duong thang: {gia_duong_thang:.4f}")
print(f"Rung ngau nhien: {gia_rung:.4f}")

# Nhận xét:
# Đường hồi quy tuyến tính có MAE thấp nhất là 0.3123,
# tiếp theo là rừng ngẫu nhiên với MAE 0.3128.
# Cây hồi quy sâu 3 có MAE cao nhất là 0.5193.
# Vì vậy, đường hồi quy tuyến tính cho kết quả dự đoán tốt nhất
# trên tập kiểm tra trong thí nghiệm này.
# Với căn nhà 250 m2, đường thẳng dự đoán giá 20.0672,
# còn cây hồi quy và rừng dự đoán lần lượt là 8.8950 và 9.0214.
# Rừng ngẫu nhiên không ngoại suy như đường hồi quy tuyến tính,
# vì mỗi cây dự đoán bằng giá trị trung bình của các mẫu trong lá.
# Rừng lấy trung bình dự đoán của 200 cây nên dự đoán vẫn nằm
# trong phạm vi giá mục tiêu của dữ liệu huấn luyện.
