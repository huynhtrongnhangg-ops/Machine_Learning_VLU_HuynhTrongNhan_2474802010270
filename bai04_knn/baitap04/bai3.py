# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("data/sinh_vien_mo_rong.csv")
#Chia cột chi_tieu từ nghìn đồng sang triệu đồng
df["chi_tieu"] = df["chi_tieu"] / 1000
#Lấy ba cột dữ liệu và nhãn
X = df[["gio_on", "diem_giua_ky", "chi_tieu"]]
y = df["qua_mon"]
#Chia dữ liệu giống mục 3
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)
#Huấn luyện KNN không chuẩn hóa, k = 7
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_tr, y_tr)
#Dự đoán và tính accuracy
y_du_doan = knn.predict(X_te)
acc = accuracy_score(y_te, y_du_doan)
print(f"Accuracy sau khi doi don vi: {acc:.4f}")
print("Accuracy truoc khi doi don vi: 0.5333")
print(f"Chenh lech: {acc - 0.5333:+.4f}")
