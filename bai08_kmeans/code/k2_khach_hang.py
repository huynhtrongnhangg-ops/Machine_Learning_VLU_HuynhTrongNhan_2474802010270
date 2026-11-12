# -*- coding: utf-8 -*-
"""Buoc 2: phan khuc khach hang bang K-Means."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/khach_hang.csv")
print("Kich thuoc bang:", df.shape)
print(df.describe().round(1).to_string())
print()

# Chuan hoa hai cot dung de phan cum
X = df[["thu_nhap", "diem_chi_tieu"]]
sc = StandardScaler().fit(X)
Xs = sc.transform(X)

# Phan cum thanh 5 nhom
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs)
df["cum"] = km.labels_

# Doi tam cum ve don vi goc de de doc
tam = pd.DataFrame(
    sc.inverse_transform(km.cluster_centers_),
    columns=["thu_nhap", "diem_chi_tieu"]
)
tam["so_khach"] = np.bincount(km.labels_)
tam["tuoi_tb"] = df.groupby("cum")["tuoi"].mean()
print("Tam cum, don vi goc:")
print(tam.round(1).to_string())
print()
print(f"Inertia: {km.inertia_:.2f}")
print()

# Xep hai khach moi vao cum bang bo chuan hoa cu
moi = pd.DataFrame({
    "thu_nhap": [29.0, 9.0],
    "diem_chi_tieu": [85, 75]
})
cum_moi = km.predict(sc.transform(moi))
for i in range(len(moi)):
    print(
        f"Khach moi {i + 1}: thu nhap {moi.iloc[i, 0]}, "
        f"chi tieu {moi.iloc[i, 1]} -> cum {cum_moi[i]}"
    )
