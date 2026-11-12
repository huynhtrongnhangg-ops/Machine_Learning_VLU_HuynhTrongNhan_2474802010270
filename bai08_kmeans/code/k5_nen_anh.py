# -*- coding: utf-8 -*-
"""Buoc 5: nen mau anh bang K-Means."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

Path("figures").mkdir(exist_ok=True)

# Tai anh mau co san trong scikit-learn
anh = load_sample_image("china.jpg")
cao, rong, _ = anh.shape
diem = anh.reshape(-1, 3) / 255.0
so_mau = len(np.unique(anh.reshape(-1, 3), axis=0))
print(f"Anh {cao} x {rong}, co {cao * rong} diem anh, {so_mau} mau")
print()

# Hoc tren 5000 diem anh duoc chon ngau nhien
rng = np.random.default_rng(0)
mau = diem[rng.choice(len(diem), 5000, replace=False)]

anh_nen = {}
print(" K    so bit moi diem   nho hon   sai lech trung binh")
for k in [4, 16, 64]:
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(mau)
    nhan = km.predict(diem)
    moi = km.cluster_centers_[nhan].reshape(cao, rong, 3)
    anh_nen[k] = np.clip(moi, 0, 1)

    bit = int(np.ceil(np.log2(k)))
    sai = np.abs(moi.reshape(-1, 3) - diem).mean() * 255
    print(f" {k:2d}         {bit}                  {24 / bit:4.1f} lan      {sai:5.2f} muc sang")

# Luu hinh so sanh anh goc va anh nen
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(anh)
axes[0].set_title("Anh goc")
axes[1].imshow(anh_nen[16])
axes[1].set_title("Chi con 16 mau")
axes[2].imshow(anh_nen[4])
axes[2].set_title("Chi con 4 mau")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig("figures/k5_nen_anh.png", dpi=150)
plt.close(fig)
print("Da luu hinh: figures/k5_nen_anh.png")
