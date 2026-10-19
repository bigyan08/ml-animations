"""The exact, unregularized two-weight logistic objective shown in the film."""

import numpy as np

SEED = 12
LEARNING_RATE = 1.5


def dataset():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(80, 2))
    x[:, 1] = 0.55 * x[:, 0] + 0.8 * x[:, 1]
    probabilities = sigmoid(x @ np.array([1.5, -1.1]))
    y = (rng.random(len(x)) < probabilities).astype(float)
    return x, y


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -np.asarray(z)))


def loss(w):
    x, y = dataset()
    z = x @ np.asarray(w)
    if z.ndim > 1:
        y = y[:, None]
    return np.mean(np.logaddexp(0, z) - y * z, axis=0)


def gradient(w):
    x, y = dataset()
    return x.T @ (sigmoid(x @ w) - y) / len(y)


def trajectory(steps=80):
    weights = [np.array([-2.5, 2.1])]
    for _ in range(steps):
        weights.append(weights[-1] - LEARNING_RATE * gradient(weights[-1]))
    return np.array(weights)


def verify():
    """Check calculus independently and confirm the displayed descent is valid."""
    path = trajectory()
    eps = 1e-5
    for w in path[::10]:
        numerical = np.array([
            (loss(w + eps * e) - loss(w - eps * e)) / (2 * eps)
            for e in np.eye(2)
        ])
        np.testing.assert_allclose(gradient(w), numerical, atol=1e-8)
    values = np.array([loss(w) for w in path])
    np.testing.assert_allclose(loss(path.T), values)
    assert np.all(np.diff(values) < 0)
    x, _ = dataset()
    smoothness_bound = np.linalg.eigvalsh(x.T @ x / (4 * len(x)))[-1]
    assert LEARNING_RATE < 1 / smoothness_bound
    assert np.linalg.norm(gradient(path[-1])) < 0.0002
    print(f"Gradient verified; loss {values[0]:.6f} -> {values[-1]:.6f}")
    print(f"Learning rate {LEARNING_RATE}; 1/L = {1 / smoothness_bound:.3f}")


if __name__ == "__main__":
    verify()
