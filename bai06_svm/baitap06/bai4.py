# -*- coding: utf-8 -*-
"""Bài 4: dò lưới mịn quanh cặp C và gamma tốt nhất."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

# Đọc dữ liệu hai vầng trăng
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

# Chia dữ liệu giống tài liệu
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Dò lưới mịn quanh C = 10, gamma = 1
luoi_moi = {
    "C": [3, 10, 30],
    "gamma": [0.3, 1, 3]
}
tim_kiem = GridSearchCV(
    SVC(kernel="rbf"),
    luoi_moi,
    cv=5
)
tim_kiem.fit(X_tr, y_tr)

# In bảng điểm kiểm định chéo
bang = pd.DataFrame(tim_kiem.cv_results_["params"])
bang["diem"] = tim_kiem.cv_results_["mean_test_score"]

print("Diem kiem dinh cheo cua luoi min")
print(
    bang.pivot(index="C", columns="gamma", values="diem")
    .round(4)
    .to_string()
)
print()

print("Cap tot nhat:", tim_kiem.best_params_)
print(f"Diem kiem dinh cheo tot nhat: {tim_kiem.best_score_:.4f}")
print(f"Accuracy tren tap kiem tra: {tim_kiem.score(X_te, y_te):.4f}")
print()

# So sánh với kết quả dò lưới ban đầu trong tài liệu
print("Moc luoi ban dau: CV = 0.8711, accuracy test = 0.9200")
print(
    "Nhan xet: doi chieu diem CV va accuracy test de danh gia "
    "luoi min co cai thien hay khong."
)
