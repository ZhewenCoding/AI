import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow.keras as keras
# import autograd.numpy as np

# N = 100
# x = np.random.rand(N,)
# eps = np.random.randn(len(x),)    #randn是标准正态分布 是 ~N(0,1)
# eps = np.random.normal(size=(len(x),), scale=0.1)
# t = 2 * x + 0.5 + eps
# plt.plot(x,t,'r.')
# plt.show()

N = 100 # Number of training data points
x = np.random.uniform(size=(N,))
# len(x)
print(len(x))
# eps = np.random.normal(size=(len(x),), scale=0.1)  #scale表示正态分布的delta  (delta=0.1)
# t = 2.0 * x + 0.5 + eps
# plt.plot(x, t, 'b.')    #r red     b blue
# plt.show()

# print(tf.__version__)

# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
