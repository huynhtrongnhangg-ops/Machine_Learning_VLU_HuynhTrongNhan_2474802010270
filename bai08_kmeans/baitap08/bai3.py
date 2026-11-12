# -*- coding: utf-8 -*-
"""Bai 3: so sanh khoi tao random va k-means++."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/khach_hang.csv")
X = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])

dem = {"random": 0, "k-means++": 0}
inertia_tot = {"random": [], "k-means++": []}

for cach in ["random", "k-means++"]:
    for rs in range(50):
        km = KMeans(
            n_clusters=5,
            init=cach,
            n_init=1,
            random_state=rs
        ).fit(X)
        inertia_tot[cach].append(km.inertia_)
        if abs(km.inertia_ - 23.0) < 0.05:
            dem[cach] += 1

print("So lan dat inertia xap xi 23.0 tren 50 lan:")
for cach in ["random", "k-means++"]:
    print(
        f" {cach:10s}: {dem[cach]} lan, "
        f"inertia nho nhat = {min(inertia_tot[cach]):.2f}, "
        f"lon nhat = {max(inertia_tot[cach]):.2f}"
    )

if dem["k-means++"] > dem["random"]:
    print("Ket luan: k-means++ dang tin cay hon trong thi nghiem nay.")
elif dem["k-means++"] < dem["random"]:
    print("Ket luan: random dat muc tieu nhieu lan hon trong thi nghiem nay.")
else:
    print("Ket luan: hai cach dat so lan bang nhau trong thi nghiem nay.")
