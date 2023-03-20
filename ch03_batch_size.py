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

x, t = get_data()
network = init_network()

# Execution Time
st = time.time()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


# Execution Time
et = time.time()
res = et - st
print('Execution time using batch_size 100:', res, 'seconds')


# Execution Time
st = time.time()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352

# Execution Time
et = time.time()

# Execution Time
res = et - st
print('Execution time using batch_size 1:', res, 'seconds')


# Execution Time
# st = time.time()
# et = time.time()
# res = et - st
# print('CPU Execution time using batch_size:', res, 'seconds')
# Accuracy:0.9352
# Execution time using batch_size 100: 0.024999618530273438 seconds
# Accuracy:0.9352
# Execution time using batch_size 1: 0.4790008068084717 seconds


# CPU Execution Time
# st = time.process_time()
# et = time.process_time()
# res = et - st
# print('CPU Execution time using batch_size:', res, 'seconds')
# Accuracy:0.9352
# CPU Execution time using batch_size 100: 0.1875 seconds
# Accuracy:0.9352
# CPU Execution time using batch_size 1: 0.578125 seconds

