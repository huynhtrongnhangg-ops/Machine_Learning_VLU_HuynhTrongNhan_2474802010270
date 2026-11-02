
# -*- coding: utf-8 -*-
"""Bài tập 3: so sánh Gini và Entropy."""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Đọc dữ liệu khách vay
df = pd.read_csv("data/khach_vay.csv")

X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

# Chia dữ liệu thành tập học và tập kiểm tra
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

# Tìm độ sâu tốt nhất bằng kiểm định chéo
do_sau_tot_nhat = None
diem_cv_tot_nhat = -1

for do_sau in range(1, 16):
    cay = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=do_sau,
        random_state=0
    )

    diem_cv = cross_val_score(
        cay,
        X_tr,
        y_tr,
        cv=5,
        scoring="accuracy"
    ).mean()

    if diem_cv > diem_cv_tot_nhat:
        diem_cv_tot_nhat = diem_cv
        do_sau_tot_nhat = do_sau

# Huấn luyện cây entropy với độ sâu tốt nhất
cay_entropy = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=do_sau_tot_nhat,
    random_state=0
)

cay_entropy.fit(X_tr, y_tr)

# Đánh giá accuracy trên tập kiểm tra
accuracy_test = cay_entropy.score(X_te, y_te)

print("Ket qua Entropy")
print(f"Do sau tot nhat: {do_sau_tot_nhat}")
print(f"Accuracy CV: {diem_cv_tot_nhat:.4f}")
print(f"Accuracy tap kiem tra: {accuracy_test:.4f}")


# Nhận xét:
# Entropy chọn độ sâu tốt nhất là 5, với accuracy tập kiểm tra đạt 0.8467.
# So với Gini ở mục 4, độ sâu tốt nhất vẫn là 5 và accuracy tập kiểm tra
# gần tương đương, nên việc đổi thước đo không làm kết quả khác đi nhiều.
