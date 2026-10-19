"""Small numerical examples for the neuron and affine-composition explanation."""
import numpy as np

INPUTS = np.array([0.8, 0.4])
WEIGHTS = np.array([[1.5, -1.0], [0.5, 1.0], [-1.0, 0.5], [1.0, 0.5]])
BIASES = np.array([-0.2, 0.0, 0.1, -0.1])


def layer_outputs():
    return np.maximum(0, WEIGHTS @ INPUTS + BIASES)


def verify():
    np.testing.assert_allclose(WEIGHTS[0] @ INPUTS + BIASES[0], 0.6)
    np.testing.assert_allclose(layer_outputs(), [0.6, 0.8, 0.0, 0.9])
    second = np.array([[0.3, -0.2, 0.8, 0.1], [-0.1, 0.7, 0.4, 0.3]])
    bias = np.array([0.1, -0.3])
    composed = second @ (WEIGHTS @ INPUTS + BIASES) + bias
    collapsed = (second @ WEIGHTS) @ INPUTS + second @ BIASES + bias
    np.testing.assert_allclose(composed, collapsed)
    assert not np.allclose(second @ layer_outputs() + bias, collapsed)
    print('PASS: neuron arithmetic, four layer responses, affine collapse, nonlinear contrast')


if __name__ == '__main__':
    verify()
