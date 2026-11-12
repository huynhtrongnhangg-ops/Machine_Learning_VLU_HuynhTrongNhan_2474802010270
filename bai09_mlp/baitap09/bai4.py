# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=17, stratify=y)
print("Toc do hoc   So vong   Loss cuoi   Accuracy")
for toc_do in [.0001, .001, .01, .1]:
    mang = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=toc_do, max_iter=1000, random_state=0))
    mang.fit(X_tr, y_tr)
    mlp = mang[-1]
    print(f"{toc_do:10g} {mlp.n_iter_:9d} {mlp.loss_curve_[-1]:11.5f} {accuracy_score(y_te, mang.predict(X_te)):.4f}")
    plt.plot(range(1, len(mlp.loss_curve_)+1), mlp.loss_curve_, label=f"lr={toc_do:g}")
plt.xlabel("Vong hoc")
plt.ylabel("Loss")
plt.title("So sanh toc do hoc MLP")
plt.legend()
plt.grid(alpha=.25)
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/toc_do_hoc.png", dpi=160, bbox_inches="tight")
plt.close()
print("Da luu: figures/toc_do_hoc.png")
print("Nhan xet: toc do qua nho hoi tu cham; qua lon co the dao dong hoac khong hoi tu.")
