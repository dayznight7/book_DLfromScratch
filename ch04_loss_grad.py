import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
import pickle
from ch03_function import sigmoid, softmax
from ch03_MNIST import get_data, init_network, predict
import time


# 오차제곱합: 모든 결과가 정확할수록 0에 가까워짐
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# res = sum_squares_error(np.array(y), np.array(t))
# res2 = sum_squares_error(np.array(y2), np.array(t))
# print(res, res2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def cross_entropy_error2(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 교차 엔트로피 오차: 정답 자리의 출력이 전체값 결정
def cross_entropy_error3(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# res3 = cross_entropy_error3(np.array(y), np.array(t))
# res4 = cross_entropy_error3(np.array(y2), np.array(t))
# print(res3, res4)


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


def my_numerical_grad(f, x):
    # x = [ [x0,x1,x2,x3,...], [y0,y1,...], [z...], [w...], ... ]
    # xT = [ [x0,y0,z0,w0,...], [x1,y1,z1,w1,...], [x2,y2,z2,w2,...], ... ]
    # grad: [ [df/dx0,df/dy0,df/dz0,df/dw0], [df/dx1,...], [...], ... ]
    # gradT: [ [df/dx0, df/dx1, df/dx2, ...], [df/dy...], [df/dz...], [df/dw...] ]
    h = 1e-4
    xT = x.T
    grad = np.zeros_like(xT)
    for i in range(x[0].size):
        for j in range(xT[0].size):
            tmp_val = xT[i][j]
            xT[i][j] = tmp_val + h
            fxh1 = f(xT[i])
            xT[i][j] = tmp_val - h
            fxh2 = f(xT[i])
            grad[i][j] = (fxh1 - fxh2) / (2*h)
            xT[i][j] = tmp_val
    return grad.T


def test_f(x):
    return x[0]**2 + x[1]**2


# gradient 시각화
# x0 = np.arange(-2, 2.5, 0.25)
# x1 = np.arange(-2, 2.5, 0.25)
# X, Y = np.meshgrid(x0, x1)
# xv = np.array([X.flatten(), Y.flatten()])
# grad = my_numerical_grad(test_f, xv)
# plt.quiver(X, Y, grad[0], grad[1])
# plt.show()


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
    return x


# 경사하강법
# init_x = np.array([-3.0, 4.0])
# ans = gradient_descent(test_f, init_x=init_x, lr=0.1, step_num=100)
# print(ans)

