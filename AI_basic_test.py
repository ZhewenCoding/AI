#记录一些网上学习的小模块 方便练习和调用
import math

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#表示行矩阵和列矩阵
    #行矩阵
x_np = np.array([[1,2,3,4]])
y_np = np.array([[1],[2],[3],[4]])  #一个[]表示一行 几个[]表示几列
print(x_np)
print('------------')
print(y_np)

#输入X输出Y转换成固定类型矩阵
X.shape = (n_x,m)
Y.shape = (1,m)

#dJ(w)/dw 编程用dw表示

#全零矩阵
dw = np.zeros((n_x,1))

#For循环算z
z = 0
for i in range(n_x):
    z += w[i] * x[i]
    pass
z+=b


#向量化 算z vectorization
z = np.dot(w.T,x) + b
#b输入实数 python会将其broadcasting成矩阵(1,m)
    #验证np.dot   一维数组算的是点积   多维算的是矩阵积 即(m,n)*(n,m)→(m,m)
a = np.array([[1,2,3],[2,3,4]])
a_T = a.T   #.T 转置  沿着对称轴反转矩阵
print(a)
print(a_T)
b = np.array([[2,2],[2,2],[2,2]])
print('b:',b)
print(np.dot(a,b))

#单个数求指数
b = math.exp(2)
print(b)

#矩阵全体元素求指数    其它  np.log()  np.abs()    np.maximum(v,0) 所有元素和0相比的最大值
x = np.array([[2,3,4],[3,2,1],[1,1,0]])
y = np.exp(x)
print(y)
print(np.maximum(x,2))

#验证broadcasting
x = np.array([[1,2,3]])
print(x+5)
print(x*2)

#将权重weight初始化为很小的随机数    注意乘上0.01(或类似的小数) 否则可能梯度下降很慢 学习慢
w_1 = np.random.randn((2,2)) * 0.01
b_1 = np.zeros((2,1))
w_2 = np.random.randn((1,2)) * 0.01
b_2 = np.zeros((2,1))

#Regularization parameter (one of huper parameter)
#用lambd表示 因为lambda已内置于python

#生成特定区间的随机数(矩阵) 例如[-4,0]
r = -4 * np.random.rand(3,3)
# print(r)
#alpha以r为指数
alpha = 10 ** r
# print(alpha)

#Batch normalization 代码
# tf.nn.batch_normalization

#Convolution
# python: conv_forward
# tensorflow: tf.nn.conv2d
# keras: Conv2D

