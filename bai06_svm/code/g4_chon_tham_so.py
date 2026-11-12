# -*- coding: utf-8 -*-
"""Bước 4: xem ảnh hưởng của gamma và dò lưới C, gamma."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

# Đọc dữ liệu hai vầng trăng
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Thử các giá trị gamma
print("gamma   tap hoc   tap kiem tra   so vecto HT")
for gamma in [0.1, 1, 10, 100]:
    mo_hinh = SVC(kernel="rbf", gamma=gamma).fit(X_tr, y_tr)
    acc_hoc = mo_hinh.score(X_tr, y_tr)
    acc_kiem_tra = mo_hinh.score(X_te, y_te)

    print(
        f" {gamma:5} {acc_hoc:.4f}      {acc_kiem_tra:.4f}"
        f"          {len(mo_hinh.support_)}"
    )
print()

# Dò lưới bằng kiểm định chéo 5 phần trên tập học
luoi_tham_so = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.01, 0.1, 1, 10]
}
tim_kiem = GridSearchCV(
    SVC(kernel="rbf"),
    luoi_tham_so,
    cv=5
)
tim_kiem.fit(X_tr, y_tr)

# In bảng điểm của 16 cặp tham số
bang = pd.DataFrame(tim_kiem.cv_results_["params"])
bang["diem"] = tim_kiem.cv_results_["mean_test_score"]

print("Diem kiem dinh cheo, hang la C, cot la gamma")
print(
    bang.pivot(index="C", columns="gamma", values="diem")
    .round(4)
    .to_string()
)
print()

print("Cap tot nhat:", tim_kiem.best_params_)
print(f"Diem kiem dinh cheo: {tim_kiem.best_score_:.4f}")
print(
    f"Accuracy tren tap kiem tra: "
    f"{tim_kiem.score(X_te, y_te):.4f}"
)
