# -*- coding: utf-8 -*-
"""Bai 5: dung DBSCAN phan cum du lieu hai vang trang."""

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]].to_numpy()

# Thu ba gia tri eps theo yeu cau
print(" eps    so cum    so diem nhieu")
for e in [0.2, 0.3, 0.4]:
    db = DBSCAN(eps=e, min_samples=5).fit(X)
    nhan = db.labels_
    so_cum = len(set(nhan) - {-1})
    so_nhieu = int((nhan == -1).sum())
    print(f" {e:.1f}      {so_cum:2d}          {so_nhieu:3d}")

print("\nNhan xet:")
print("DBSCAN co the tim cum co hinh dang cong va tach hai vang trang.")
print("eps nho co the tao nhieu diem nhieu; eps lon co the gom cac cum lai.")
