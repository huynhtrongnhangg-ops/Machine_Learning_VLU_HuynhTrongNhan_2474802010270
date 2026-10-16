# Bài 4: Tu viet bo loc khong dung thu vien
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv("data/tin_nhan.csv")
# Chia du lieu thanh tap hoc va tap kiem tra
X_train, X_test, y_train, y_test = train_test_split(
    df["noi_dung"], df["la_rac"],
    test_size=0.25, random_state=17, stratify=df["la_rac"]
)
# Tao tu dien chi tu tap hoc
tu_dien = sorted(set(
    tu
    for tin in X_train
    for tu in tin.split()
))
V = len(tu_dien)
vi_tri_tu = {tu: i for i, tu in enumerate(tu_dien)}
# Dem so lan xuat hien cua tung tu trong moi lop
dem_tu = {0: Counter(), 1: Counter()}
tong_tu = {0: 0, 1: 0}
so_tin = {0: 0, 1: 0}
for tin, nhan in zip(X_train, y_train):
    so_tin[nhan] += 1
    for tu in tin.split():
        if tu in vi_tri_tu:
            dem_tu[nhan][tu] += 1
            tong_tu[nhan] += 1
# Tinh log xac suat cua tung tu voi Laplace smoothing
alpha = 1.0
log_xac_suat = {0: {}, 1: {}}
for nhan in [0, 1]:
    mau_so = tong_tu[nhan] + alpha * V
    for tu in tu_dien:
        tu_so = dem_tu[nhan][tu] + alpha
        log_xac_suat[nhan][tu] = np.log(tu_so / mau_so)
# Tinh log xac suat tien nghiem cua hai lop
log_tien_nghiem = {
    nhan: np.log(so_tin[nhan] / len(X_train))
    for nhan in [0, 1]
}
# Du doan bang tong log xac suat
def du_doan(tin):
    diem = {
        nhan: log_tien_nghiem[nhan]
        for nhan in [0, 1]
    }
    for tu in tin.split():
        if tu in vi_tri_tu:
            for nhan in [0, 1]:
                diem[nhan] += log_xac_suat[nhan][tu]

    return 1 if diem[1] > diem[0] else 0
y_du_doan = [du_doan(tin) for tin in X_test]
acc_tu_viet = accuracy_score(y_test, y_du_doan)
# So sanh voi MultinomialNB tren cung tap du lieu
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X_train_nb = vec.fit_transform(X_train)
X_test_nb = vec.transform(X_test)
mo_hinh = MultinomialNB(alpha=1.0)
mo_hinh.fit(X_train_nb, y_train)
y_pred_nb = mo_hinh.predict(X_test_nb)
acc_thu_vien = accuracy_score(y_test, y_pred_nb)
print(f"Accuracy bo loc tu viet: {acc_tu_viet:.4f}")
print(f"Accuracy MultinomialNB: {acc_thu_vien:.4f}")
print(f"Do chenh lech: {abs(acc_tu_viet - acc_thu_vien):.4f}")
# Nhan xet:
# Bo loc tu viet va MultinomialNB deu su dung Naive Bayes
# voi Laplace smoothing alpha = 1.0.
# Tinh tong log xac suat giup tranh tran so khi nhan nhieu
# xac suat nho voi nhau.
# Hai mo hinh co accuracy gan nhau vi cung dem tu,
# tinh xac suat theo lop va su dung cung tap du lieu.
