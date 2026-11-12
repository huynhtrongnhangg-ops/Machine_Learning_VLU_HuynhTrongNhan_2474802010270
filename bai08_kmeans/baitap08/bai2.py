# -*- coding: utf-8 -*-
"""Bai 2: phan cum tren ca ba cot va chon K bang silhouette."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/khach_hang.csv")
cot = ["tuoi", "thu_nhap", "diem_chi_tieu"]
sc = StandardScaler()
X = sc.fit_transform(df[cot])

print("Silhouette khi dung ca ba cot:")
ket_qua = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    sil = silhouette_score(X, km.labels_)
    ket_qua.append((k, sil, km))
    print(f" K = {k:2d}, silhouette = {sil:.4f}")

k_tot, sil_tot, km_tot = max(ket_qua, key=lambda x: x[1])
tam_goc = pd.DataFrame(
    sc.inverse_transform(km_tot.cluster_centers_), columns=cot
)
tam_goc["so_khach"] = pd.Series(km_tot.labels_).value_counts().sort_index().values

print()
print(f"K tot nhat: {k_tot}, silhouette = {sil_tot:.4f}")
print("Tam cum theo don vi goc:")
print(tam_goc.round(1).to_string(index=False))
