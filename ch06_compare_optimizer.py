import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# one_hot_label=True: (0 1 0 ... 0)
# normalize=True: 784 pixel data 0.0~1.0, False: 0~255
# flatten=True: 784 1dim

# x_train: (60000, 784)
# t_train: (60000, 10)
# x_test: (10000, 784)
# t_test: (10000, 10)


# 학습 횟수
# 총 데이터 크기: 60000개
# 미니배치 크기: 100개
iters_num = 2000
train_size = x_train.shape[0]
batch_size = 100

# 학습 방법
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()
optimizers['RMSprop'] = RMSprop()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []


# 학습 iters_num 만큼 시작
for i in range(iters_num):
    # batch_냨
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)


markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D", "RMSprop": "p"}
learning_interval = 100
x = np.arange(iters_num/learning_interval)
for key in optimizers.keys():
    plt.plot(x, train_loss[key][0::learning_interval], marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()

