import pandas as pd

df = pd.read_csv("data/gia_nha.csv")

nhom_lon = df[df["dien_tich"] > 100]

print("So can co dien tich lon hon 100 m2:", len(nhom_lon))
print("Gia trung binh cua nhom:", round(nhom_lon["gia"].mean(), 3), "ty dong")