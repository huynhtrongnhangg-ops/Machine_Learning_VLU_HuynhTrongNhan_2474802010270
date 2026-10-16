# Bài 6: Lop hiem va xac suat tien nghiem
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score
df = pd.read_csv("data/tin_nhan.csv")
# Lay toan bo tin thuong va 20 tin rac dau tien
tin_thuong = df[df["la_rac"] == 0]
tin_rac = df[df["la_rac"] == 1].head(20)
df_moi = pd.concat([tin_thuong, tin_rac], ignore_index=True)
X = df_moi["noi_dung"]
y = df_moi["la_rac"]
# Chia du lieu thanh tap hoc va tap kiem tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)
mo_hinh = MultinomialNB(alpha=1.0)
mo_hinh.fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
print("So tin trong bo du lieu moi:", len(df_moi))
print(f"Accuracy = {acc:.4f}")
print(f"Recall lop rac = {rec:.4f}")
# Nhan xet:
# Du lieu co 180 tin thuong nhung chi co 20 tin rac,
# nen mo hinh co the thien ve du doan tin thuong.
# Accuracy cao chua chac bo loc tot vi co the bo sot nhieu tin rac.
# Recall lop rac cho biet ty le tin rac thuc su duoc phat hien.