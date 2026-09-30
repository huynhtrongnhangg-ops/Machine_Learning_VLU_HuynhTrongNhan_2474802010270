# -*- coding: utf-8 -*-
"""Bài tập 1: Thống kê theo nhóm."""
import pandas as pd
df = pd.read_csv("data/sinh_vien.csv")
# 7 điểm trở lên
nhom_tren_7 = df[df["diem_giua_ky"] >= 7]
# Nhóm còn lại
nhom_duoi_7 = df[df["diem_giua_ky"] < 7]
so_ban_tren_7 = len(nhom_tren_7)
ty_le_qua_tren_7 = nhom_tren_7["qua_mon"].mean()
ty_le_qua_duoi_7 = nhom_duoi_7["qua_mon"].mean()
print("THONG KE THEO NHOM")
print(f"So ban co diem giua ky >= 7: {so_ban_tren_7}")
print(f"Ty le qua mon cua nhom >= 7: {ty_le_qua_tren_7:.4f}")
print(f"Ty le qua mon cua nhom < 7: {ty_le_qua_duoi_7:.4f}")
print()
print("Nhan xet:")
if ty_le_qua_tren_7 > ty_le_qua_duoi_7:
    print("Diem giua ky co phan biet duoc hai nhom ve ty le qua mon.")
else:
    print("Diem giua ky khong phan biet ro hai nhom ve ty le qua mon.")