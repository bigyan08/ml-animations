"""One normalized house example for a single supervised learning update."""

import numpy as np

SIZE = 1.0  # Thousands of square feet.
ACTUAL_PRICE = 2.0  # Hundreds of thousands of dollars.
INITIAL_W = 1.0
LEARNING_RATE = 0.5


def prediction(w):
    return np.asarray(w) * SIZE


def loss(w):
    return 0.5 * (prediction(w) - ACTUAL_PRICE) ** 2


def gradient(w):
    return (prediction(w) - ACTUAL_PRICE) * SIZE


def update(w):
    return np.asarray(w) - LEARNING_RATE * gradient(w)


def verify():
    """Check each displayed value from the stated model and loss."""
    assert np.isclose(prediction(INITIAL_W), 1.0)
    assert np.isclose(loss(INITIAL_W), 0.5)
    assert np.isclose(gradient(INITIAL_W), -1.0)
    next_w = float(update(INITIAL_W))
    assert np.isclose(next_w, 1.5)
    assert np.isclose(prediction(next_w), 1.5)
    assert np.isclose(loss(next_w), 0.125)
    assert loss(next_w) < loss(INITIAL_W)
    print("Verified: w 1→1.5; prediction 1→1.5; loss 0.5→0.125.")


if __name__ == "__main__":
    verify()
