# -*- coding: utf-8 -*-
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=17, stratify=y)
# Dò lưới chỉ trên tập học; dùng ba phần để giảm thời gian chạy
mang = make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000, random_state=0))
luoi = {"mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50,50)], "mlpclassifier__alpha": [.0001, .01, 1.]}
tim = GridSearchCV(mang, luoi, cv=3, n_jobs=-1)
tim.fit(X_tr, y_tr)
acc = accuracy_score(y_te, tim.predict(X_te))
svm = SVC().fit(X_tr, y_tr)
acc_svm = accuracy_score(y_te, svm.predict(X_te))
print("Tham so tot nhat:", tim.best_params_)
print(f"CV tot nhat: {tim.best_score_:.4f}")
print(f"MLP tap kiem tra: {acc:.4f}")
print(f"SVM RBF tap kiem tra: {acc_svm:.4f}")
print("MLP vuot SVM:", "Co" if acc > acc_svm else "Khong")
