# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
#Đọc dữ liệu giá nhà
df = pd.read_csv("data/gia_nha.csv")
#Lấy diện tích và giá nhà
X = df[["dien_tich"]]
y = df["gia"]
#Chia tập học và tập kiểm tra giống bài e5
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#Chọn k bằng kiểm định chéo 5 phần trên tập học
diem = {}
for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k)
    #Điểm trả về là MAE mang dấu âm
    scores = cross_val_score(
        knn,
        X_tr,
        y_tr,
        cv=5,
        scoring="neg_mean_absolute_error"
    )
    #Đổi về MAE dương để dễ đọc
    diem[k] = -scores.mean()
#Chọn k có MAE kiểm định chéo nhỏ nhất
k_tot = min(diem, key=diem.get)
print("MAE kiem dinh cheo theo k:")
for k in range(1, 21):
    print(f"k = {k:2d}: MAE = {diem[k]:.4f}")
print()
print(f"k tot nhat = {k_tot}")
print(f"MAE kiem dinh cheo nho nhat = {diem[k_tot]:.4f}")
#Huấn luyện KNN với k tốt nhất
knn_tot = KNeighborsRegressor(n_neighbors=k_tot)
knn_tot.fit(X_tr, y_tr)
#Huấn luyện mô hình đường thẳng
duong = LinearRegression()
duong.fit(X_tr, y_tr)
#So sánh MAE trên tập kiểm tra
mae_knn = mean_absolute_error(y_te, knn_tot.predict(X_te))
mae_duong = mean_absolute_error(y_te, duong.predict(X_te))
print()
print("So sanh MAE tren tap kiem tra (ty dong)")
print(f"KNN, k = {k_tot}: {mae_knn:.4f}")
print(f"Duong thang:       {mae_duong:.4f}")
if mae_knn < mae_duong:
    print("Ket luan: KNN co MAE thap hon duong thang.")
elif mae_knn > mae_duong:
    print("Ket luan: KNN co MAE cao hon duong thang.")
else:
    print("Ket luan: Hai mo hinh co MAE bang nhau.")
