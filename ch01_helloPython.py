import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

arr1 = np.array([[1,2,3],[4,5,6]])
print(arr1)
print(arr1[0][2])

x = np.arange(0,6,0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, linestyle="--", label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.legend()
plt.show()
