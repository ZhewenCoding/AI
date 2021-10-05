import numpy as np

class NN():
    def __init__(self):
        #随机数种子生成随机数 每次生成的初始权重w相同
        np.random.seed(0)
        #生成3*1的随机权重矩阵
        self.synaptic_weights = 2 * np.random.random((3,1)) - 1 #生成(-1,1)中的随机数 突触权重

    #激活函数sigmoid "去线性化"
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    # 激活函数sigmoid的导数 调整权重
    def sigmoid_derivative(self,x):
        return x * (1-x)

    #对节点输出值激活 前向传播并激活
    def think(self,inputs):
        inputs = inputs.astype(float)   #转换成float格式
        output = self.sigmoid(np.dot(inputs,self.synaptic_weights)) #矩阵乘法x*w 再通过sigmoid激活
        return output

    #定义训练过程 (输入值 输出值 训练轮数)
    def train(self,training_inputs,training_outputs,train_steps):
        for step in range(train_steps):
            output = self.think(training_inputs)
            #计算训练数据真实值和网络预测值的训练误差
            error = training_outputs - output
            #模拟反向传播 调整误差 误差加权导数
            adjustments = np.dot(training_inputs.T,error * self.sigmoid_derivative(output))
            #微调权重参数
            self.synaptic_weights = self.synaptic_weights + adjustments
            pass
        pass
    pass

if __name__ == '__main__':
    neural_Network = NN()
    print('初始权重：%s'%neural_Network.synaptic_weights)
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1],
                                [0,1,0],
                                [0,0,0]])
    training_outputs = np.array([[0,1,1,0,0,0]]).T
    neural_Network.train(training_inputs,training_outputs,200000)
    print('训练后权重：%s'%neural_Network.synaptic_weights)
    user_input_one = str(input('输入第一个值：'))
    user_input_two = str(input('输入第二个值：'))
    user_input_three = str(input('输入第三个值：'))
    print('输入值为：',user_input_one,user_input_two,user_input_three)
    new_output = neural_Network.think(np.array([user_input_one,user_input_two,user_input_three]))
    print('预测输出值：%s'%new_output)

