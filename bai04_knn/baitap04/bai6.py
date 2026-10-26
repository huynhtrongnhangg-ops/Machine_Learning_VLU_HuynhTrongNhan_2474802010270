# -*- coding: utf-8 -*-
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
#Tao bo sinh so ngau nhien
rng = np.random.default_rng(0)
#Cac kich thuoc du lieu can thu nghiem
kich_thuoc = [1000, 10000, 100000]
#In tieu de bang ket qua
print("So mau     Fit (giay)     Predict (giay)")
for n in kich_thuoc:
    #Tao du lieu hoc gom n mau va 2 dac trung
    X_hoc = rng.random((n, 2))
    #Tao nhan ngau nhien cho cac mau
    y_hoc = rng.integers(0, 2, size=n)
    #Tao 1000 mau moi de du doan
    X_moi = rng.random((1000, 2))
    #Khoi tao mo hinh KNN voi k = 5
    knn = KNeighborsClassifier(n_neighbors=5)
    #Do thoi gian huan luyen
    bat_dau = time.perf_counter()
    knn.fit(X_hoc, y_hoc)
    thoi_gian_fit = time.perf_counter() - bat_dau
    #Do thoi gian du doan 1000 mau moi
    bat_dau = time.perf_counter()
    knn.predict(X_moi)
    thoi_gian_predict = time.perf_counter() - bat_dau
    #In ket qua
    print(
        f"{n:6d}     "
        f"{thoi_gian_fit:10.6f}     "
        f"{thoi_gian_predict:14.6f}"
    )
