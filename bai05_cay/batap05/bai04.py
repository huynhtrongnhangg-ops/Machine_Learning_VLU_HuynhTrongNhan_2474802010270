
# -*- coding: utf-8 -*-
"""Bài tập 4: chọn min_samples_leaf bằng kiểm định chéo."""

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

# Các giá trị số khách tối thiểu ở mỗi lá cần thử
cac_gia_tri = [1, 2, 5, 10, 20, 40]

gia_tri_tot_nhat = None
accuracy_cv_tot_nhat = -1

# Thử từng giá trị bằng kiểm định chéo 5 phần
for so_khach in cac_gia_tri:
    cay = DecisionTreeClassifier(
        min_samples_leaf=so_khach,
        random_state=0
    )

    diem_cv = cross_val_score(
        cay,
        X_tr,
        y_tr,
        cv=5,
        scoring="accuracy"
    ).mean()

    print(
        f"Min samples leaf: {so_khach}, "
        f"Accuracy CV: {diem_cv:.4f}"
    )

    if diem_cv > accuracy_cv_tot_nhat:
        accuracy_cv_tot_nhat = diem_cv
        gia_tri_tot_nhat = so_khach

# Huấn luyện cây với giá trị tốt nhất
cay_tot_nhat = DecisionTreeClassifier(
    min_samples_leaf=gia_tri_tot_nhat,
    random_state=0
)

cay_tot_nhat.fit(X_tr, y_tr)

# Đánh giá một lần trên tập kiểm tra
accuracy_test = cay_tot_nhat.score(X_te, y_te)

print()
print("Ket qua cat tia")
print(f"Min samples leaf tot nhat: {gia_tri_tot_nhat}")
print(f"Accuracy CV tot nhat: {accuracy_cv_tot_nhat:.4f}")
print(f"Accuracy tap kiem tra: {accuracy_test:.4f}")


# Nhận xét:
# Giá trị min_samples_leaf tốt nhất là 10, với accuracy CV đạt 0.8089.
# Accuracy trên tập kiểm tra đạt 0.8467.
# Khi tăng số khách tối thiểu ở mỗi lá, cây được hạn chế chia quá nhỏ,
# giúp giảm độ phức tạp và có thể hạn chế học thuộc dữ liệu.
# Tuy nhiên, nếu giá trị quá lớn như 40, accuracy CV giảm còn 0.7133.
# Cần so sánh accuracy tập kiểm tra với cây sâu 5 ở mục 4
# để xác định cách nào cho kết quả tốt hơn trên bộ dữ liệu này.
