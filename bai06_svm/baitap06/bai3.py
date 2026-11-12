# -*- coding: utf-8 -*-
"""Bài 3: so sánh kernel đa thức với các bậc 2, 3 và 4."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Đọc dữ liệu hai vầng trăng
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

# Chia dữ liệu giống tài liệu
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Dò C riêng cho từng bậc đa thức
ket_qua = []
for bac in [2, 3, 4]:
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel="poly", degree=bac)
    )

    luoi = {"svc__C": [0.1, 1, 10, 100]}
    tim_kiem = GridSearchCV(pipeline, luoi, cv=5)
    tim_kiem.fit(X_tr, y_tr)

    acc_test = tim_kiem.score(X_te, y_te)
    ket_qua.append({
        "Bac": bac,
        "C tot nhat": tim_kiem.best_params_["svc__C"],
        "Accuracy CV": tim_kiem.best_score_,
        "Accuracy kiem tra": acc_test
    })

    print(f"Kernel da thuc bac {bac}")
    print(" C tot nhat:", tim_kiem.best_params_["svc__C"])
    print(f" Accuracy CV: {tim_kiem.best_score_:.4f}")
    print(f" Accuracy tap kiem tra: {acc_test:.4f}")
    print()

# Tổng hợp kết quả
bang = pd.DataFrame(ket_qua)
print("Bang so sanh cac bac kernel da thuc")
print(bang.to_string(index=False))

# Mốc tham khảo RBF ở mục 5 của tài liệu
print()
print("Moc tham khao kernel RBF muc 5: CV = 0.8711, test = 0.9200")
