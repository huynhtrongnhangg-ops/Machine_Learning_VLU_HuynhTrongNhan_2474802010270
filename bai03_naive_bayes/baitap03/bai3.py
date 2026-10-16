# Bài 3: Doi tham so lam tron
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv("data/tin_nhan.csv")
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
for alpha in [0.001, 0.1, 1.0, 10.0, 100.0]:
    mo_hinh = MultinomialNB(alpha=alpha)
    mo_hinh.fit(X_train, y_train)
    y_pred = mo_hinh.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    print(f"Alpha = {alpha}")
    print(f"Accuracy = {acc:.4f}")
    print(f"Precision = {pre:.4f}")
    print(f"Recall = {rec:.4f}")
    print()
# Nhan xet: Alpha qua lon lam xac suat cac tu o hai lop xich lai gan nhau,
# nen mo hinh kho phan biet tin rac va tin thuong dua tren cac tu.
