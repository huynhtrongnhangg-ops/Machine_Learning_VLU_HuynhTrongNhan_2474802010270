# -*- coding: utf-8 -*-
"""Bai 6: phan cum chu so viet tay bang K-Means."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

# Nap anh chu so, khong dua nhan that vao qua trinh phan cum
du_lieu = load_digits()
X = du_lieu.data
nhan_that = du_lieu.target

km = KMeans(n_clusters=10, n_init=10, random_state=0)
cum = km.fit_predict(X)

# Chi sau khi phan cum moi doi chieu voi nhan that
bang = pd.crosstab(
    pd.Series(nhan_that, name="Nhan that"),
    pd.Series(cum, name="Cum K-Means")
)
print("Bang doi chieu nhan that va cum:")
print(bang.to_string())

print("\nCum chua nhieu mau nhat cua tung chu so:")
for so in range(10):
    cum_tot = bang.loc[so].idxmax()
    ti_le = bang.loc[so].max() / bang.loc[so].sum()
    print(f" Chu so {so}: cum {cum_tot}, ty le tap trung {ti_le:.3f}")

print("\nMoi hang cho biet mot chu so that duoc phan bo vao cac cum nhu the nao.")
print("Cot co nhieu chu so khac nhau cho thay cac chu so bi gom chung.")
