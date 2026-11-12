# -*- coding: utf-8 -*-
import numpy as np

# Hai nơ-ron ẩn đo độ chênh lệch theo hai chiều
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
h1 = np.maximum(0, X[:,0] - X[:,1])
h2 = np.maximum(0, X[:,1] - X[:,0])
y = h1 + h2
print("x1 x2  h1 h2  y")
for x, a, b, c in zip(X, h1, h2, y):
    print(f"{x[0]:.0f}  {x[1]:.0f}   {a:.0f}  {b:.0f}   {c:.0f}")
print("Giai thich: hai dau vao giong nhau thi h1=h2=0; khac nhau thi mot no-ron bang 1, nen y=1.")
