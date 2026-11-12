# -*- coding: utf-8 -*-
"""Buoc 4: minh hoa ba cai bay cua K-Means."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

Path("figures").mkdir(exist_ok=True)

df = pd.read_csv("data/khach_hang.csv")
X = df[["thu_nhap", "diem_chi_tieu"]]
Xs = StandardScaler().fit_transform(X)

# Bay 1: thu nhap co don vi lon hon va khong chuan hoa
X_nghin = X.copy()
X_nghin["thu_nhap"] = X_nghin["thu_nhap"] * 1000
km_tho = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X_nghin)
tam_tho = km_tho.cluster_centers_.copy()
tam_tho[:, 0] = tam_tho[:, 0] / 1000
print("Bay 1: khong chuan hoa, tam cum (thu nhap trieu, chi tieu):")
for t in tam_tho[np.argsort(tam_tho[:, 0])]:
    print(f" ({t[0]:5.1f}, {t[1]:5.1f})")
print()

# Bay 2: chi khoi tao ngau nhien mot lan
ket = []
for rs in range(20):
    km = KMeans(
        n_clusters=5, init="random", n_init=1, random_state=rs
    ).fit(Xs)
    ket.append(round(km.inertia_, 1))
print("Bay 2: inertia cua 20 lan khoi tao ngau nhien:")
ket = sorted(ket)
print(" ", ket[:10])
print(" ", ket[10:])
tot = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs)
print(f" Voi n_init=10: inertia {tot.inertia_:.1f}")
print()

# Bay 3: K-Means tren du lieu hai vang trang
tr = pd.read_csv("data/hai_vang_trang.csv")
km_tr = KMeans(n_clusters=2, n_init=10, random_state=0)
nhan = km_tr.fit_predict(tr[["x1", "x2"]])
khop = (nhan == tr["nhan"]).mean()
ty_le_khop = max(khop, 1 - khop)
print(f"Bay 3: hai vang trang, ty le khop nhan that {ty_le_khop:.4f}")

# Ve so sanh phan cum hai vang trang
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].scatter(tr["x1"], tr["x2"], c=tr["nhan"], s=15)
axes[0].set_title("Nhan that")
axes[1].scatter(tr["x1"], tr["x2"], c=nhan, s=15)
axes[1].scatter(
    km_tr.cluster_centers_[:, 0], km_tr.cluster_centers_[:, 1],
    marker="X", s=140
)
axes[1].set_title("K-Means voi K = 2")
for ax in axes:
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
fig.tight_layout()
fig.savefig("figures/k4_hai_vang_trang.png", dpi=150)
plt.close(fig)
