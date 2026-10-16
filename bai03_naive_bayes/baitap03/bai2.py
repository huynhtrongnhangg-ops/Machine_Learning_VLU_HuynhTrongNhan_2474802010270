# Bài 2: Tu nao to cao tin rac manh nhat
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv("data/tin_nhan.csv")
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]
mo_hinh = MultinomialNB(alpha=1.0)
mo_hinh.fit(X, y)
#Tinh hieu log xac suat giua lop rac va lop thuong
tu_dien = vec.get_feature_names_out()
hieu = mo_hinh.feature_log_prob_[1] - mo_hinh.feature_log_prob_[0]
bang_tu = pd.DataFrame({
    "tu": tu_dien,
    "hieu": hieu
})
#Hieu lon nhat la tu dac trung cho tin rac
print("5 tu to cao tin rac manh nhat:")
print(bang_tu.sort_values("hieu", ascending=False).head(5))
print()
#Hieu nho nhat la tu dac trung cho tin thuong
print("5 tu cho thay tin thuong manh nhat:")
print(bang_tu.sort_values("hieu", ascending=True).head(5))
