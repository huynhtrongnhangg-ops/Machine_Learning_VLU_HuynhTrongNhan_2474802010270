# -*- coding: utf-8 -*-
"""Bài 2: in classification_report cho SVM nhận dạng 10 chữ số."""

from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Nạp bộ dữ liệu chữ số viết tay như bài 6
X, y = load_digits(return_X_y=True)

# Chia dữ liệu giống bài 6
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Huấn luyện SVM RBF
svm = SVC(kernel="rbf")
svm.fit(X_tr, y_tr)
du = svm.predict(X_te)

# In báo cáo cho đủ 10 chữ số
print("Classification report cho 10 chu so:")
print(classification_report(y_te, du, labels=list(range(10)), digits=4))

# Tìm chữ số có recall và precision thấp nhất
bao_cao = classification_report(
    y_te, du, labels=list(range(10)), output_dict=True, zero_division=0
)
recall_thap = min(range(10), key=lambda so: bao_cao[str(so)]["recall"])
precision_thap = min(range(10), key=lambda so: bao_cao[str(so)]["precision"])

print("Chu so co recall thap nhat:", recall_thap,
      f"({bao_cao[str(recall_thap)]['recall']:.4f})")
print("Chu so co precision thap nhat:", precision_thap,
      f"({bao_cao[str(precision_thap)]['precision']:.4f})")
print("Macro avg tinh deu 10 chu so; weighted avg tinh theo so mau moi chu so.")
print("Hai gia tri gan nhau vi bo digits co so mau giua cac chu so kha dong deu.")
