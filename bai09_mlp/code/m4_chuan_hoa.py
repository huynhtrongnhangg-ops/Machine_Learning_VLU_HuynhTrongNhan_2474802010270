# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sinh_vien_mo_rong.csv")
X, y = df[["gio_on", "diem_giua_ky", "chi_tieu"]], df["qua_mon"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=17, stratify=y)
tho = MLPClassifier(hidden_layer_sizes=(20,), max_iter=3000, random_state=0).fit(X_tr, y_tr)
chuan = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(20,), max_iter=3000, random_state=0)).fit(X_tr, y_tr)
print(f"Doan bua ca lop qua : {y_te.mean():.4f}")
print(f"Khong chuan hoa     : {accuracy_score(y_te, tho.predict(X_te)):.4f}")
print(f"Co chuan hoa        : {accuracy_score(y_te, chuan.predict(X_te)):.4f}")
du = tho.predict(X_te)
print("Khong chuan hoa, so ban doan qua:", du.sum(), "tren", len(du))
