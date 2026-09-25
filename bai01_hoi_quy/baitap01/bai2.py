import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/gia_nha.csv")

plt.scatter(df["so_phong"], df["gia"])

plt.xlabel("So phong")
plt.ylabel("Gia (ty dong)")
plt.title("Moi quan he giua so phong va gia")

plt.savefig("bai2.png", dpi=150)

plt.show()

print("So phong tang thi gia can ho co xu huong tang, nhung cac diem khong hoan toan nam tren mot duong thang.")