# -*- coding: utf-8 -*-
"""Bài 5: confusion matrix và lưu ảnh chữ số SVM dự đoán sai."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Tạo thư mục lưu ảnh nếu chưa có
thu_muc_anh = Path("figures/bai5_chu_so_sai")
thu_muc_anh.mkdir(parents=True, exist_ok=True)

# Nạp dữ liệu chữ số
X, y = load_digits(return_X_y=True)

# Chia dữ liệu giống mục 6
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Huấn luyện SVM RBF mặc định
svm = SVC(kernel="rbf")
svm.fit(X_tr, y_tr)
du_doan = svm.predict(X_te)

# Tạo confusion matrix: hàng là nhãn thật, cột là nhãn dự đoán
ma_tran_nham_lan = confusion_matrix(y_te, du_doan, labels=np.arange(10))
print("Confusion matrix (hang: nhan that, cot: nhan du doan)")
print(ma_tran_nham_lan)

# Tìm và in các cặp chữ số bị nhầm
sai = np.flatnonzero(du_doan != y_te)
print()
print("So anh du doan sai:", len(sai))
print("Cac cap chu so bi nham (that -> doan):")

cap_nham_lan = {}
for i in sai:
    cap = (int(y_te[i]), int(du_doan[i]))
    cap_nham_lan[cap] = cap_nham_lan.get(cap, 0) + 1

for cap, so_lan in sorted(cap_nham_lan.items()):
    print(f" {cap[0]} -> {cap[1]}: {so_lan} lan")

# Vẽ và lưu riêng từng ảnh dự đoán sai
for thu_tu, i in enumerate(sai, start=1):
    plt.figure(figsize=(3, 3))
    plt.imshow(X_te[i].reshape(8, 8), cmap="gray_r")
    plt.title(f"That: {y_te[i]} - Doan: {du_doan[i]}")
    plt.axis("off")

    ten_anh = (
        f"anh_{thu_tu:02d}_that_{y_te[i]}_doan_{du_doan[i]}.png"
    )
    duong_dan = thu_muc_anh / ten_anh
    plt.savefig(duong_dan, bbox_inches="tight", dpi=150)
    plt.close()

print()
print(f"Da luu {len(sai)} anh vao: {thu_muc_anh}")
