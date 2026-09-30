# -*- coding: utf-8 -*-
"""Bài tập 6: Đổi lớp dương rồi chấm lại."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
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
precision_lop_0 = precision_score(
    y_test, y_pred, pos_label=0
)
recall_lop_0 = recall_score(
    y_test, y_pred, pos_label=0
)
print("Danh gia lop 0 - Rot mon")
print(f"Precision lop 0 = {precision_lop_0:.4f}")
print(f"Recall lop 0 = {recall_lop_0:.4f}")
print()
print("So sanh voi lop 1 - Qua mon")
print("Precision lop 1 = 0.8500")
print("Recall lop 1 = 0.9444")

# Ở các bài trước, lớp 1 (qua môn) được chọn làm lớp dương.
# Vì vậy Precision và Recall được tính dựa trên khả năng dự đoán lớp qua môn.

# Ở bài này, pos_label=0 nên lớp 0 (rớt môn) được chọn làm lớp dương.
# Khi đó Precision cho biết trong những trường hợp mô hình dự đoán rớt,
# có bao nhiêu trường hợp thực sự rớt.
# Recall cho biết trong những sinh viên thực sự rớt,
# mô hình phát hiện đúng được bao nhiêu sinh viên.

# Kết quả Precision và Recall của lớp 0 khác lớp 1
# vì đối tượng được đánh giá đã thay đổi từ lớp qua môn sang lớp rớt môn.
# Dữ liệu và mô hình không thay đổi, chỉ thay đổi lớp được xem là lớp dương.

# nhận xét:
# Khi đổi lớp dương từ 1 sang 0, Precision và Recall cũng thay đổi.
# Điều này cho thấy Precision và Recall phụ thuộc vào lớp dương
# mà ta đang quan tâm.