import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys, os
# sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
import pickle
from ch03_function import sigmoid, softmax


def test_load_mnist():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False)
    print(x_train.shape)  # (60000, 784) 28*28 이미지 6만개
    print(t_train.shape)  # (60000,) 정답 6만개
    print(x_test.shape)  # (10000, 784)
    print(t_test.shape)  # (10000,)

    img = x_train[0]
    label = t_train[0]
    print(label)
    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)
    img_show(img)


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# test_load_mnist()


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False) # normalize=True : 0~255 -> 0~1
    return x_test, t_test


def init_network():
    with open("ch03/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = sigmoid(a3)

    return y


def run_MNIST():
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352

