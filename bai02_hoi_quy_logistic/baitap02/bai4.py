# -*- coding: utf-8 -*-
"""Bài tập 4: Tự tính bốn thước đo."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)
# Thứ tự bốn ô là TN, FP, FN, TP
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("Ma tran nham lan")
print(f" TN = {tn:2d} doan rot, that su rot")
print(f" FP = {fp:2d} doan qua, that ra rot")
print(f" FN = {fn:2d} doan rot, that ra qua")
print(f" TP = {tp:2d} doan qua, that su qua")
print()
# Tự tính bốn thước đo bằng công thức
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print("Tu tinh bang cong thuc:")
print(f"Accuracy = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall = {recall:.4f}")
print(f"F1 = {f1:.4f}")

# Accuracy là tỷ lệ số dự đoán đúng trên tổng số dự đoán.
# Precision cho biết trong những trường hợp mô hình dự đoán là qua môn,
# có bao nhiêu trường hợp thực sự qua môn.
# Recall cho biết trong những sinh viên thực sự qua môn,
# mô hình dự đoán đúng được bao nhiêu sinh viên.
# F1 là chỉ số kết hợp giữa Precision và Recall.

# Với bài này:
# TN = 9: mô hình dự đoán rớt và thực tế cũng rớt.
# FP = 3: mô hình dự đoán qua nhưng thực tế là rớt.
# FN = 1: mô hình dự đoán rớt nhưng thực tế là qua.
# TP = 17: mô hình dự đoán qua và thực tế cũng qua.

# Accuracy dùng cả TP và TN để đánh giá số dự đoán đúng.
# Precision bị ảnh hưởng bởi FP vì có những trường hợp dự đoán qua nhưng thực tế rớt.
# Recall bị ảnh hưởng bởi FN vì có những trường hợp thực sự qua nhưng mô hình dự đoán rớt.
# F1 kết hợp Precision và Recall nên dùng để xem mô hình cân bằng giữa hai chỉ số này như thế nào.

# nhận xét:
# Mô hình có Accuracy = 0.8667, nghĩa là dự đoán đúng khoảng 86.67% số trường hợp.
# Precision = 0.8500, nghĩa là trong các trường hợp được dự đoán qua môn,
# có 85% thực sự qua môn.
# Recall = 0.9444, nghĩa là mô hình phát hiện đúng 94.44% số sinh viên thực sự qua môn.
# F1 = 0.8947, cho thấy Precision và Recall của mô hình khá cân bằng.