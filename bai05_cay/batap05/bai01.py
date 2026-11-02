
# -*- coding: utf-8 -*-
"""Bài tập 1: vẽ cây năm lá và cây sâu 2."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Đọc dữ liệu khách vay
df = pd.read_csv("data/khach_vay.csv")

X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

# Chia dữ liệu giống các bài trước
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Bài 1: cây có năm lá
cay_nam_la = DecisionTreeClassifier(
    max_leaf_nodes=5,
    random_state=0
)

cay_nam_la.fit(X_tr, y_tr)

print("CAY NAM LA")
print(f"Do sau: {cay_nam_la.get_depth()}")
print(f"So la: {cay_nam_la.get_n_leaves()}")

# Vẽ cây năm lá
plt.figure(figsize=(16, 9))

plot_tree(
    cay_nam_la,
    feature_names=list(X.columns),
    class_names=["Tre han", "Dung han"],
    filled=True,
    rounded=True
)

plt.title("Cay quyet dinh nam la")
plt.savefig("cay_nam_la.png", dpi=150, bbox_inches="tight")
plt.show()

print("Da luu cay_nam_la.png")
print()


# Bài 2: cây có độ sâu bằng 2
cay_sau_2 = DecisionTreeClassifier(
    max_depth=2,
    random_state=0
)

cay_sau_2.fit(X_tr, y_tr)

print("CAY SAU 2")
print(f"Do sau: {cay_sau_2.get_depth()}")
print(f"So la: {cay_sau_2.get_n_leaves()}")

# Vẽ cây sâu 2
plt.figure(figsize=(12, 7))

plot_tree(
    cay_sau_2,
    feature_names=list(X.columns),
    class_names=["Tre han", "Dung han"],
    filled=True,
    rounded=True
)

plt.title("Cay quyet dinh do sau 2")
plt.savefig("cay_sau_2.png", dpi=150, bbox_inches="tight")
plt.show()

print("Da luu cay_sau_2.png")
print()


# Kiểm tra hai lá bên trái của cây sâu 2
print("KIEM TRA HAI LA BEN TRAI")

# Lấy hai nút con bên trái của nút gốc
nut_goc = 0

nut_trai = cay_sau_2.tree_.children_left[nut_goc]

nut_trai_trai = cay_sau_2.tree_.children_left[nut_trai]
nut_trai_phai = cay_sau_2.tree_.children_right[nut_trai]

# Lấy lớp dự đoán của từng lá
lop_trai = cay_sau_2.tree_.value[nut_trai_trai][0]
lop_phai = cay_sau_2.tree_.value[nut_trai_phai][0]

du_doan_trai = lop_trai.argmax()
du_doan_phai = lop_phai.argmax()

print("La ben trai:", du_doan_trai)
print("La ben phai:", du_doan_phai)

if du_doan_trai == du_doan_phai:
    print("Hai la co cung ket luan")
else:
    print("Hai la co ket luan khac nhau")

# Nhận xét:
# Cây quyết định năm lá giúp phân chia dữ liệu khách vay thành các nhóm
# dựa trên những đặc trưng như thu nhập và tình trạng việc làm.
# Khi giới hạn độ sâu bằng 2, hai lá bên trái có thể có cùng kết luận
# dự đoán nhưng vẫn được chia thành hai nút.
# Nguyên nhân là cây quyết định lựa chọn cách chia dựa trên mức giảm Gini,
# nhằm làm cho các nhóm dữ liệu thuần hơn, chứ không chỉ dựa vào
# việc hai lá có cùng nhãn dự đoán hay không.