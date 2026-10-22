"""Exact toy data and simultaneous full-batch gradient descent."""
import numpy as np
X = np.array([1., 2., 3., 4.])
Y = np.array([50., 60., 70., 80.])

def loss(w, b):
    return float(np.mean((w * X + b - Y) ** 2))

def train(steps=500, rate=.05):
    w = b = 0.
    history = [(w, b, loss(w, b))]
    for _ in range(steps):
        error = w * X + b - Y
        dw, db = 2 * np.mean(error * X), 2 * np.mean(error)
        w, b = w - rate * dw, b - rate * db
        history.append((w, b, loss(w, b)))
    return np.asarray(history)
