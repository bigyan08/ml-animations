"""Numerical examples behind the scalar update and regression animation."""
import numpy as np

X = np.array([0.4, 0.8, 1.2, 1.7, 2.1, 2.5, 2.9])
Y = 0.8 + 1.3 * X + np.array([0.12, -0.16, 0.09, -0.11, 0.18, -0.12, 0.05])
DESIGN = np.column_stack([X, np.ones_like(X)])


def scalar_loss(p):
    return (np.asarray(p) - 10) ** 2


def regression_loss(weights):
    return float(np.mean((DESIGN @ weights - Y) ** 2))


def fit_path(steps=60):
    weights = np.array([0.3, 0.1])
    path = [weights.copy()]
    for _ in range(steps):
        gradient = 2 * DESIGN.T @ (DESIGN @ weights - Y) / len(Y)
        weights = weights - 0.08 * gradient
        path.append(weights.copy())
    return np.array(path)


def verify():
    gradient = 2 * (7 - 10)
    next_p = 7 - 0.1 * gradient
    assert np.isclose(next_p, 7.6)
    assert np.isclose(scalar_loss(7), 9)
    assert np.isclose(scalar_loss(next_p), 5.76)
    losses = np.array([regression_loss(w) for w in fit_path()])
    assert np.all(np.diff(losses) <= 1e-10)
    assert losses[-1] < 0.05
    assert np.isclose(0.7 + 0.3, 1)
    print(f'Scalar: 7→7.6, loss 9→5.76. Regression MSE: {losses[0]:.4f}→{losses[-1]:.4f}.')


if __name__ == '__main__':
    verify()
