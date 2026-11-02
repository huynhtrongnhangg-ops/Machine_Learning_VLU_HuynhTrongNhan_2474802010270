
# -*- coding: utf-8 -*-
"""Bài tập 5: so sánh Feature Importance và Permutation Importance."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

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

# Tạo cột ngẫu nhiên cho cả tập học và tập kiểm tra
rng = np.random.default_rng(0)

X_tr_nn = X_tr.copy()
X_te_nn = X_te.copy()

X_tr_nn["so_ngau_nhien"] = rng.random(len(X_tr_nn))
X_te_nn["so_ngau_nhien"] = rng.random(len(X_te_nn))

# Huấn luyện rừng ngẫu nhiên
rung = RandomForestClassifier(
    n_estimators=200,
    random_state=0
)

rung.fit(X_tr_nn, y_tr)

# Độ quan trọng theo feature_importances_
do_quan_trong = pd.Series(
    rung.feature_importances_,
    index=X_tr_nn.columns
)

print("Feature importance")
print(do_quan_trong.sort_values(ascending=False).round(4).to_string())

# Permutation Importance trên tập kiểm tra
ket_qua_xao_tron = permutation_importance(
    rung,
    X_te_nn,
    y_te,
    n_repeats=20,
    random_state=0,
    scoring="accuracy"
)

do_quan_trong_xao_tron = pd.Series(
    ket_qua_xao_tron.importances_mean,
    index=X_te_nn.columns
)

print()
print("Permutation importance")
print(
    do_quan_trong_xao_tron
    .sort_values(ascending=False)
    .round(4)
    .to_string()
)

# Xác định thứ hạng của cột ngẫu nhiên
thu_hang_xao_tron = (
    do_quan_trong_xao_tron
    .rank(ascending=False, method="min")
    ["so_ngau_nhien"]
)

thu_hang_feature = (
    do_quan_trong
    .rank(ascending=False, method="min")
    ["so_ngau_nhien"]
)

print()
print("Thu hang cot ngau nhien")
print(f"Permutation importance: {int(thu_hang_xao_tron)}")
print(f"Feature importance: {int(thu_hang_feature)}")


# Nhận xét:
# Theo feature_importances_, cột so_ngau_nhien đứng thứ 3
# với độ quan trọng 0.1499, nhưng theo permutation importance
# cột này đứng thứ 5 với giá trị chỉ 0.0017.
# Nguyên nhân là feature_importances_ đo mức giảm độ hỗn tạp
# tại các nút mà đặc trưng được sử dụng để chia cây,
# nên cột ngẫu nhiên vẫn có thể nhận độ quan trọng cao.
# Trong khi đó, permutation importance đo mức giảm accuracy
# khi xáo trộn cột trên tập kiểm tra.
# Vì xáo trộn cột ngẫu nhiên gần như không làm giảm accuracy,
# cột này hầu như không có giá trị dự đoán thực sự.
# Do đó, permutation importance phản ánh rõ hơn ảnh hưởng
# thực tế của cột ngẫu nhiên đến khả năng dự đoán trên tập kiểm tra.
