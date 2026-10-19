"""One-dimensional quadratic used to demonstrate an excessive learning rate."""

import numpy as np


def loss(w):
    return 0.5 * np.asarray(w) ** 2


def gradient(w):
    return np.asarray(w)


def update(w, learning_rate):
    return np.asarray(w) - learning_rate * gradient(w)


def verify():
    """Check the two displayed steps directly from the stated objective."""
    start = 1.0
    large = float(update(start, 2.4))
    moderate = float(update(start, 1.5))
    assert np.isclose(large, -1.4)
    assert np.isclose(loss(start), 0.5)
    assert np.isclose(loss(large), 0.98)
    assert loss(large) > loss(start)
    assert np.isclose(moderate, -0.5)
    assert np.isclose(loss(moderate), 0.125)
    assert loss(moderate) < loss(start)
    print("Verified: η=2.4 raises loss 0.50→0.98; η=1.5 lowers it to 0.125.")


if __name__ == "__main__":
    verify()
