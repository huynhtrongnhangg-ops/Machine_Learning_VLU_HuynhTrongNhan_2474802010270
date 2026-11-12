# -*- coding: utf-8 -*-
"""Buoc 3: chon so cum K bang khuỷu tay va silhouette."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/khach_hang.csv")
Xs = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])

cac_k = list(range(1, 11))
inertias = []
silhouettes = []
truoc = None

print(" K     inertia   giam so voi K-1   silhouette")
for k in cac_k:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    inertias.append(km.inertia_)
    giam = "" if truoc is None else f"{truoc - km.inertia_:8.1f}"
    sil = "" if k == 1 else f"{silhouette_score(Xs, km.labels_):.3f}"
    silhouettes.append(None if k == 1 else silhouette_score(Xs, km.labels_))
    print(f" {k:2d}    {km.inertia_:7.1f}   {giam:>15s}   {sil:>10s}")
    truoc = km.inertia_

# Ve hai bieu do chon K
Path("figures").mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(cac_k, inertias, marker="o")
axes[0].set_title("Phuong phap khuyu tay")
axes[0].set_xlabel("So cum K")
axes[0].set_ylabel("Inertia")
axes[0].grid(True, alpha=0.3)

axes[1].plot(cac_k[1:], silhouettes[1:], marker="o")
axes[1].set_title("Chi so silhouette")
axes[1].set_xlabel("So cum K")
axes[1].set_ylabel("Silhouette")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/k3_chon_k.png", dpi=150)
plt.close(fig)
print("Da luu bieu do: figures/k3_chon_k.png")
