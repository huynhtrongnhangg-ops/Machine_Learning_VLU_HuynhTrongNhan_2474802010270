
# -*- coding: utf-8 -*-

"""
Bước 1: Tự tính khoảng cách và tự tìm láng giềng, chưa dùng thư viện KNN.
Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import numpy as np
import pandas as pd

# Ba bạn: A và C học gần như y hệt nhau, B học khác hẳn
ba_ban = {
    "A": np.array([5.0, 3.0, 4000.0]),
    "B": np.array([25.0, 9.0, 4100.0]),
    "C": np.array([6.0, 3.5, 7000.0]),
}


# Hàm tính khoảng cách Euclid giữa hai điểm
def kc(u, v):
    return np.sqrt(((u - v) ** 2).sum())


# Tính khoảng cách khi chưa chuẩn hóa
print("Khoang cach KHI CHUA chuan hoa")
print(f" d(A, B) = {kc(ba_ban['A'], ba_ban['B']):8.1f}")
print(f" d(A, C) = {kc(ba_ban['A'], ba_ban['C']):8.1f}")
print()


# Đọc dữ liệu sinh viên từ file CSV
df = pd.read_csv("data/sinh_vien_mo_rong.csv")

# Lấy ba cột dữ liệu
X = df[["gio_on", "diem_giua_ky", "chi_tieu"]].to_numpy(dtype=float)

# Tính trung bình và độ lệch chuẩn của từng cột
tb = X.mean(axis=0)
dl = X.std(axis=0)

# In độ lệch chuẩn
print("Do lech chuan cua ba cot:")
print(f" gio_on {dl[0]:.2f} diem_giua_ky {dl[1]:.2f}"
      f" chi_tieu {dl[2]:.0f}")
print()


# Chuẩn hóa ba bạn bằng trung bình và độ lệch chuẩn của dữ liệu
chuan = {
    ten: (v - tb) / dl
    for ten, v in ba_ban.items()
}

# Tính khoảng cách sau khi chuẩn hóa
print("Khoang cach SAU KHI chuan hoa")
print(f" d(A, B) = {kc(chuan['A'], chuan['B']):.2f}")
print(f" d(A, C) = {kc(chuan['A'], chuan['C']):.2f}")
print()


# Lấy hai cột học tập để tìm láng giềng
X2 = df[["gio_on", "diem_giua_ky"]].to_numpy(dtype=float)

# Lấy nhãn kết quả qua môn
y = df["qua_mon"].to_numpy()

# Tính trung bình và độ lệch chuẩn của hai cột
tb2 = X2.mean(axis=0)
dl2 = X2.std(axis=0)

# Tạo dữ liệu của bạn mới
ban_moi = np.array([12.0, 6.5])

# Chuẩn hóa dữ liệu sinh viên và bạn mới
X2c = (X2 - tb2) / dl2
moi_c = (ban_moi - tb2) / dl2

# Tính khoảng cách từ bạn mới đến tất cả sinh viên
khoang = np.sqrt(((X2c - moi_c) ** 2).sum(axis=1))

# Lấy vị trí của 5 sinh viên gần nhất
gan = np.argsort(khoang)[:5]

# In thông tin 5 láng giềng gần nhất
print("Ban moi: on 12 gio, diem giua ky 6.5")
print("Nam lang gieng gan nhat:")

for i in gan:
    ket = "qua" if y[i] == 1 else "rot"
    print(f" gio {X2[i, 0]:5.1f} diem {X2[i, 1]:4.1f} "
          f"khoang cach {khoang[i]:.3f} -> {ket}")

# Đếm số láng giềng qua môn
so_qua = int(y[gan].sum())

# Dự đoán theo đa số phiếu
print(f"Bo phieu: {so_qua} qua, {5 - so_qua} rot")
print("Du doan:", "qua mon" if so_qua >= 3 else "rot mon")
