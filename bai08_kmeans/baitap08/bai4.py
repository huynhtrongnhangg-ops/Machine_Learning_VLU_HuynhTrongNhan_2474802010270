# -*- coding: utf-8 -*-
"""Bai 4: nen anh 16 mau va so sanh dung luong tep."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

thu_muc = Path("figures")
thu_muc.mkdir(exist_ok=True)


def nen_anh(anh, ten):
    """Nen anh RGB thanh 16 mau, luu anh goc va anh da nen."""
    # Bo kenh alpha neu anh co bon kenh
    if anh.ndim == 3 and anh.shape[2] == 4:
        anh = anh[:, :, :3]

    # Dua du lieu ve dang RGB tu 0 den 1
    if np.issubdtype(anh.dtype, np.integer):
        anh_rgb = anh.astype(np.float64) / 255.0
    else:
        anh_rgb = np.clip(anh.astype(np.float64), 0, 1)

    cao, rong, kenh = anh_rgb.shape
    if kenh != 3:
        raise ValueError("Anh can co 3 kenh RGB.")

    diem = anh_rgb.reshape(-1, 3)
    rng = np.random.default_rng(0)
    so_mau_hoc = min(5000, len(diem))
    mau = diem[rng.choice(len(diem), so_mau_hoc, replace=False)]

    km = KMeans(n_clusters=16, n_init=4, random_state=0).fit(mau)
    nhan = km.predict(diem)
    anh_moi = km.cluster_centers_[nhan].reshape(cao, rong, 3)
    anh_moi = np.clip(anh_moi, 0, 1)

    tep_goc = thu_muc / f"{ten}_goc.png"
    tep_nen = thu_muc / f"{ten}_16_mau.png"
    plt.imsave(tep_goc, anh_rgb)
    plt.imsave(tep_nen, anh_moi)

    dung_luong_goc = tep_goc.stat().st_size
    dung_luong_nen = tep_nen.stat().st_size
    print(f"\nAnh {ten}: {cao} x {rong}")
    print(f" Dung luong anh goc: {dung_luong_goc:,} byte")
    print(f" Dung luong anh 16 mau: {dung_luong_nen:,} byte")
    print(f" Ty le dung luong: {dung_luong_goc / dung_luong_nen:.2f} lan")

    return anh_rgb, anh_moi


# Nen anh mau cua scikit-learn
anh_goc = load_sample_image("china.jpg")
goc, nen = nen_anh(anh_goc, "china")

# Neu co anh rieng, dat tai data/anh_cua_em.png de xu ly them
tep_rieng = Path("data/anh_cua_em.png")
if tep_rieng.exists():
    anh_rieng = plt.imread(tep_rieng)
    nen_anh(anh_rieng, "anh_cua_em")
else:
    print("\nChua co data/anh_cua_em.png, da bo qua phan anh rieng.")

# Luu hinh so sanh
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(goc)
axes[0].set_title("Anh goc")
axes[1].imshow(nen)
axes[1].set_title("Anh sau khi nen 16 mau")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(thu_muc / "bai4_so_sanh_anh.png", dpi=150)
plt.close(fig)
print("Da luu hinh so sanh: figures/bai4_so_sanh_anh.png")
