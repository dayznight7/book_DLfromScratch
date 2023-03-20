import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def np_random_randn(m,n,sd):
    return np.random.randn(m,n) * sd


x1 = np_random_randn(1000, 100, 1)
x2 = x1.copy()
x3 = x1.copy()
x4 = x1.copy()

node_num = 100
hidden_layer_size = 5
activations1 = {}
activations2 = {}
activations3 = {}
activations4 = {}

for i in range(hidden_layer_size):
    if i != 0:
        x1 = activations1[i - 1]
        x2 = activations2[i - 1]
        x3 = activations3[i - 1]
        x4 = activations4[i - 1]

    w1 = np_random_randn(node_num, node_num, 1)
    w2 = np_random_randn(node_num, node_num, 0.01)
    w3 = np_random_randn(node_num, node_num, 1/np.sqrt(node_num))
    w4 = np_random_randn(node_num, node_num, 2/np.sqrt(node_num))
    activations1[i] = sigmoid(np.dot(x1, w1))
    activations2[i] = sigmoid(np.dot(x2, w2))
    activations3[i] = sigmoid(np.dot(x3, w3))
    activations4[i] = ReLU(np.dot(x4, w4))


def compare_with_plot2X5(actv1, actv2):
    for i, a in actv1.items():
        plt.subplot(2, len(actv1), i + 1)
        plt.title(str(i + 1) + "-layer actv")
        plt.hist(a.flatten(), 100)

    for i, a in actv2.items():
        plt.subplot(2, len(actv2), i + len(actv2) + 1)
        plt.title(str(i + 1) + "-layer actv'")
        plt.hist(a.flatten(), 100)

    plt.show()


# actv1: avg = 0, sd = 1            Normal, 1
# actv2: avg = 0, sd = 0.01         Normal, 0.01
# actv3: avg = 0, sd = 1/(n^0.5)    Xavier
# actv4: avg = 0, sd = 2/(n^0.5)    He
compare_with_plot2X5(activations1, activations4)

