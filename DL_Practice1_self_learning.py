import numpy as np
from matplotlib import pyplot as plt
# import tensorflow as tf
#或  import matplotlib.pyplot as plt

#生成1D (100,100)集合   x_i 在[-1,1]均匀分布  导入随机数
# x_i = np.random.rand(100)
# y_i = 0.1 * x_i + x_i ** 2 + x_i ** 3

#不用随机数 用有序数组
# n_samples = 100
# x_i = np.linspace(-1, 1, n_samples) #lineal space vector
# # a = np.linspace(0,1,10)     #生成有序数组Array 即从0到1闭区间 划分为10个数据点
# y_i = 0.1 * x_i + np.power(x_i, 2) + np.power(x_i, 3)
# print(x_i,'\n',y_i)

#建立class
# class Dataset01:
#     def __init__(self,n_samples = 100):
#         self.n_samples = n_samples
#         pass
#     def load_data(self):
#         self.x_i = np.linspace(-1, 1, self.n_samples)    #this.x_i?
#         self.y_i = 0.1 * self.x_i + np.power(self.x_i, 2) + np.power(self.x_i, 3)
#         plt.figure(num='1D数组绘图',figsize=(5, 5),facecolor='pink',frameon=True)
#         #绘图 figure(num=,figsize=,dpi=,facecolor=,edgecolor=,frameon=)
#         plt.plot(x_i,y_i)   #class中函数里的x_i,y_i不会被这里调用  只能从class外被调用
#         plt.xticks(ticks=np.linspace(-1, 1, 5), labels=np.linspace(-1, 1, 5))
#         #设置刻度间隔
#         plt.show()
#         #写在类函数中并实例化后直接调用类函数的方式
#         return self.x_i,self.y_i
#         pass
# test = Dataset01()
# test.load_data()

##Dataset.load_data()     #报错  类必须先实例化才能被调用   test = Dataset() 就是实例化
# plot x_i,y_i   5x5   用pyplot.plot  xticks
# plt.figure(figsize=(5, 5))
# plt.plot(x_i,y_i)   #class中函数里的x_i,y_i不会被这里调用  只能从class外被调用
# plt.xticks(ticks=np.linspace(-1, 1, 5), labels=np.linspace(-1, 1, 5))
# plt.show()
###假如要调用写在类里面的plot怎么调用？

#生成2D Dataset   x_i服从标准正态分布 用np.random.RandomState() 或np.random.seed()
np.random.seed(42)
x_i = np.random.randn(100,2)    #x_i为2D (100,2)型矩阵
y_i = np.ones(shape=(100,))     #y_i为1D (100,1)型矩阵
# print('x_i:',x_i)
# print('y_i:',y_i)
# print('=-=-=-=-=-=-=')
y_i[np.sum(np.square(x_i),axis = -1) < 1] = 0   #np.square 获取矩阵元素的平方    #???
# print('y_i:',y_i)
# #axis=-1 -1代表倒数第一个 例 矩阵shape=[3,4,5] 取axis=-1后 shape=[3,4]
#
# #用plt.scatter()绘制散点图
plt.figure(figsize=(5,5))
plt.scatter(x_i[y_i==0,0],x_i[y_i==0,1],marker='o',c='b',label='class 0')
# x_i[y_i==0,0]     #y_i==0 逻辑运算符 结果为0False/1True?
plt.scatter(x_i[y_i==1,0],x_i[y_i==1,1],marker='v',c='r',label='class 1')
#c 颜色(b=blue r=red)     marker =标记的样式 默认为'o'
plt.legend()
plt.show()
#
# 生成矩阵X(4,4)     全1滤波器矩阵W(2,2)       计算卷积
# X = np.array([[0,1,2,3],
#               [4,5,6,7],
#               [8,9,10,11],
#               [12,13,14,15]])
# X = np.arange(0,16,1)     #返回一个有终点起点的固定步长的排列
X = np.arange(0,16,1).reshape(4,4)  #重定形成(4,4)矩阵
W = np.ones(shape=(2,2))
print(X)
print('------------------')
print(W)
X_conv = np.zeros(shape=(3,3))
print(X_conv)
#
for i in range(X_conv.shape[0]):
    for j in range(X_conv.shape[1]):
        X_conv[i,j] = np.sum(X[i:(i+2),j:(j+2)] * W)    #':'是切片操作吗??? 没有理解
        pass
    pass
#                         #未定义函数的矩阵卷积计算
# print(X_conv)
#
# def convolve(X, W):
#     (X_height, X_width) = X.shape
#     (W_height, W_width) = W.shape
#     X_conv_height, X_conv_width = X_height - W_width + 1, X_width - W_width + 1
#     X_conv = np.zeros(shape=(X_conv_height, X_conv_width))
#
#     for i in range(X_conv.shape[0]):
#         for j in range(X_conv.shape[1]):
#             X_conv[i, j] = np.sum(X[i:(i+W_height), j:(j+W_width)] * W)
#             pass
#         pass
#     return X_conv
#                         #定义函数的矩阵卷积计算


##测试seed()
# np.random.seed(0)   #一个随机数的盆 代号为第0个
# a = np.random.rand(10)
# np.random.seed(89)  #若seed()里的数字相同 a和b生成相同的随机数
#                     #若此行不写seed(0) a和b不相同——seed()只作用一次
# b = np.random.rand(10)
# print(a)
# print('\n')
# print(b)

# np.random.seed(125)
# for i in range(5):
#     print(np.random.rand(5))    #第2 3 4 5遍是在默认random下随机生成的随机数 都与第一组不同
#     pass
# print('------------------------')
# for i in range(5):
#     np.random.seed(125)
#     print(np.random.rand(5))    #五组都是seed(125)里的随机数 所以是结果相同的5组
#     pass

# 测试np.random.RandomState()
# np.random.RandomState(0)
# a = np.random.randn(5)
# np.random.RandomState(0)
# b = np.random.randn(5)
# print(a)
# print('\n')
# print(b)        #结果不同
# print('----------------------')
# rng = np.random.RandomState(0)      #RandomState()生成伪随机数 必须在rng这个变量下使用 否则结果不同 如上
# a = rng.rand(5)
# rng = np.random.RandomState(0)
# b = rng.rand(5)
# print(a)
# print('\n')
# print(b)        #结果相同
# print('----------------------')
# rng = np.random.RandomState(0)
# a = rng.randn(5)
# b = rng.randn(5)
# print(a)
# print('\n')
# print(b)        #结果不同

#测试np.square
# array_2d = np.array([[1,2,3],[4,5,6]])
# print(f'原数组：\n{array_2d}')
# array_2d_square = np.square(array_2d)
# print(f'平方后数组：\n{array_2d_square}')

#测试'[]'
# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# a_row = 2
# a_column = 0
# print('第',a_row,'行, 第',a_column,'列为'.format(a_row,a_column),a[a_row,a_column])
# print(.format(a_column))
#验证了 a[x,y]返回a矩阵(数组)第x行y列元素   回归前面问题

        #验证axis=-1的作用
#1.先生成一个shape=[3,4,5]的np.random.rand    表示包含3个4*5的矩阵
# x = np.random.rand(3,4,5)          #可以直接生成3维矩阵
# print(x)
#2.对矩阵分别按轴(0,1,-1)取最大值所在下标
#2a.按轴0取
# a = tf.math.argmax(x,axis=0)    #无法加载tensorflow的lib  先搞定tf.lib
