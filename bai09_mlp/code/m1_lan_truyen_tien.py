# -*- coding: utf-8 -*-
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# Tính một nơ-ron bằng trọng số cho trước
w = np.array([0.8, -0.5, 0.3])
x = np.array([2., 1., 3.])
b = -0.2
z = w @ x + b
print(f"Mot no-ron: z = {z:.2f}, sigmoid(z) = {sigmoid(z):.4f}")

# Mạng hai nơ-ron ẩn tự tính XOR
W1 = np.array([[1., 1.], [1., 1.]])
b1 = np.array([0., -1.])
W2 = np.array([1., -2.])
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
H = relu(X @ W1 + b1)
Y = H @ W2
print("x1 x2  h1 h2  y")
for x, h, y in zip(X, H, Y):
    print(f"{x[0]:.0f}  {x[1]:.0f}   {h[0]:.0f}  {h[1]:.0f}   {y:.0f}")
print("Tinh ca bang mot lan:", Y)
