import numpy as np
import matplotlib.pyplot as plt
# from sympy import *

#定义sigmoid函数
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    dy = y * (1-y)        #直接利用了求导结果 网上也是这样直接给出的
    return dy
    pass

# def derivative_sigmoid(x):
#     return x * (1-x)
#     pass


# x = np.linspace(-5,5,100)   #x区间为(-5,5) 划分为100个数据点
# y = sigmoid(x)
# plt.figure(figsize=(5, 5))
# plt.scatter(x, y, marker='o', c='b', label='sigmoid')
# plt.legend()
# plt.show()              #出了Sigmoid散点图

# x = np.arange(-5,5,0.1)           #array range()
x = np.linspace(-5,5,100)           #跟arange都是限制起点终点和步长
dy = sigmoid(x)
plt.plot(x,dy)
plt.show()                #出连续图

#定义单层神经网络
def forward_pass(x,w1,b1,w2,b2,y):
    z1 = np.dot(w1,x) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(w2,a1) + b2
    a2 = z2
    y = a2






# sigmoid(np.random.rand(2,3))

#对sigmoid函数求导   待解决
# def derivative_sigmoid(x):  #要求只用numpy
#     p = 1 / (1 + np.exp(-x))

#求导流程 多项式的例子
# p = np.poly1d([1,2,0,3,0,5])    #1D多项式 1x**5+2x**4+3x**2+5
# print('p:',p)
# # print(p(2))                     #x=2时多项式的值
# for i in range(1,4):
#     print(np.polyder(p,i))      #对多项式p求i阶导数
#     print(np.polyder(p,i)(1.0)) #对多项式p求i阶导数后输出x=1.0时的值
# print('-------------------------')
# for i in range(1,4):
#     print(p.deriv(i))           #效果一样 也是求导 但是是p.xxx 不是np.xxx
#     print(p.deriv(i)(1.0))
#     pass


