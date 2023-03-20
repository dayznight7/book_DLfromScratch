import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def step_function2(x):
    if x > 0:
        return 1
    else:
        return 0


def step_function(x):
    y = x > 0
    return y.astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def show_function(f):
    x = np.arange(-5.0, 5.0, 0.1)
    y = f(x)
    plt.plot(x, y)
    plt.show()


def test_array():
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    print(A)
    print(np.ndim(A))
    print(A.shape)
    B = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    print(B.shape)
    C = np.dot(A, B)
    print(C)
    D = np.dot(B, A)
    print(D)

    X = np.array([1, 2])
    W = np.array([[1, 3, 5], [2, 4, 6]])
    Y = np.dot(X, W)
    print(Y)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

