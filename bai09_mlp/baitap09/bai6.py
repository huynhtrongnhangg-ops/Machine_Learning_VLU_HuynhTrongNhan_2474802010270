# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Dùng ba đặc trưng của dữ liệu giá nhà
bang = pd.read_csv("data/gia_nha.csv")
X = bang[["dien_tich", "so_phong", "tuoi_nha"]]
y = bang["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=17)
hoi_quy = LinearRegression().fit(X_tr, y_tr)
mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0)).fit(X_tr, y_tr)
print(f"MAE hoi quy tuyen tinh: {mean_absolute_error(y_te, hoi_quy.predict(X_te)):.4f}")
print(f"MAE MLP: {mean_absolute_error(y_te, mlp.predict(X_te)):.4f}")
# Giữ số phòng và tuổi nhà bằng trung vị để so sánh công bằng
can_moi = pd.DataFrame([{"dien_tich":250, "so_phong":float(X_tr['so_phong'].median()), "tuoi_nha":float(X_tr['tuoi_nha'].median())}])
print("Gia can 250 m2 (cac cot con lai lay trung vi tap hoc):")
print(f"Hoi quy tuyen tinh: {hoi_quy.predict(can_moi)[0]:.4f}")
print(f"MLP: {mlp.predict(can_moi)[0]:.4f}")
print("Nhan xet: 250 m2 nam ngoai mien du lieu, MLP khong dam bao ngoai suy tot nhu duong thang.")
