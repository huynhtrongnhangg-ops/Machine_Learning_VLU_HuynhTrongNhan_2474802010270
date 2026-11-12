# -*- coding: utf-8 -*-
"""Bai 1: tu tinh silhouette bang numpy."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def silhouette_mot_diem(X, nhan, i):
    """Tinh a, b va silhouette cho diem thu i."""
    # Lay cac diem trong cung cum, bo chinh diem dang xet
    cung_cum = X[(nhan == nhan[i]) & (np.arange(len(X)) != i)]
    if len(cung_cum) == 0:
        return 0.0, 0.0, 0.0

    # a la khoang cach trung binh trong cung cum
    a = np.linalg.norm(cung_cum - X[i], axis=1).mean()

    # b la khoang cach trung binh toi cum gan nhat khac
    cac_cum = [k for k in np.unique(nhan) if k != nhan[i]]
    khoang_cach_cum = []
    for k in cac_cum:
        diem_k = X[nhan == k]
        khoang_cach_cum.append(
            np.linalg.norm(diem_k - X[i], axis=1).mean()
        )
    b = min(khoang_cach_cum)

    # Silhouette cua mot diem
    s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return a, b, s


df = pd.read_csv("data/khach_hang.csv")
X = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
nhan = km.labels_

gia_tri = np.array([
    silhouette_mot_diem(X, nhan, i)[2] for i in range(len(X))
])
tu_tinh = gia_tri.mean()
thu_vien = silhouette_score(X, nhan)

print(f"Silhouette tu tinh: {tu_tinh:.6f}")
print(f"Silhouette scikit-learn: {thu_vien:.6f}")
print(f"Khach co silhouette nho nhat: {gia_tri.argmin()}")
print(f"So silhouette nho nhat: {gia_tri.min():.6f}")
print("Toa do goc cua khach nay:")
print(df.loc[gia_tri.argmin(), ["thu_nhap", "diem_chi_tieu"]].to_dict())
