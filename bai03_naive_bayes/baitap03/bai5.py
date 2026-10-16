# Bài 5: Chon sai bien the Naive Bayes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
# Thu MultinomialNB tren du lieu sinh vien
mo_hinh_1 = MultinomialNB()
mo_hinh_1.fit(X_train, y_train)
y_pred_1 = mo_hinh_1.predict(X_test)
acc_1 = accuracy_score(y_test, y_pred_1)
print(f"MultinomialNB accuracy = {acc_1:.4f}")
# So sanh voi GaussianNB
mo_hinh_2 = GaussianNB()
mo_hinh_2.fit(X_train, y_train)
y_pred_2 = mo_hinh_2.predict(X_test)
acc_2 = accuracy_score(y_test, y_pred_2)
print(f"GaussianNB accuracy = {acc_2:.4f}")
# Nhan xet: MultinomialNB van chay nhung khong phu hop
# vi du lieu gio hoc va diem thi la cac gia tri lien tuc,
# khong phai du lieu dem tan suat tu nhu mo hinh mong doi.
