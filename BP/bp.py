import numpy as np
import pandas as pd


def sigmoid(v):
    return 1/(1 + np.exp(-v))


def loadData(fileName):
    """
    :param fileName: name of input file
    :return: data, value
    """
    fp = open(fileName, encoding='utf-8')
    # 存放数据向量
    data = []
    # 存放标签值
    value = []
    for line in fp:
        entry_list = line.strip().split()
        data.append([float(j) for j in entry_list[:-1]])
        # 2值 0或1
        value.append(int(entry_list[-1]))
    return data, value


def initialize(x, y, z):
    """
    :param x: input layer num
    :param y: hidden layer num
    :param z: output layer num
    :return: weight1, weight2, value1, value2
    weight1:输入层与隐层的连接权重
    weight2:隐层与输出层的连接权重
    value1:隐层阈值
    value2:输出层阈值
    """
    value1 = np.random.randint(-5, 5, (1, y)).astype(np.float64)
    value2 = np.random.randint(-5, 5, (1, z)).astype(np.float64)
    weight1 = np.random.randint(-5, 5, (x, y)).astype(np.float64)
    weight2 = np.random.randint(-5, 5, (y, z)).astype(np.float64)
    return weight1, weight2, value1, value2


def training(data, value, weight1, weight2, value1, value2):
    # x为步长
    x = 0.01
    for i in range(len(data)):
        # 输入数据
        inputset = np.mat(data[i]).astype(np.float64)
        # 数据标签
        outputset = np.mat(value[i]).astype(np.float64)
        # 隐层输入
        input1 = np.dot(inputset, weight1).astype(np.float64)
        # 隐层输出
        output2 = sigmoid(input1 - value1).astype(np.float64)
        # 输出层输入
        input2 = np.dot(output2, weight2).astype(np.float64)
        # 输出层输出
        output3 = sigmoid(input2 - value2).astype(np.float64)

        # 更新公式由矩阵计算
        a = np.multiply(output3, 1 - output3)
        g = np.multiply(a, outputset - output3)
        b = np.dot(g, np.transpose(weight2))
        c = np.multiply(output2, 1 - output2)
        e = np.multiply(b, c)

        value1_change = -x * e
        value2_change = -x * g
        weight1_change = x * np.dot(np.transpose(inputset), e)
        weight2_change = x * np.dot(np.transpose(output2), g)

        # 更新参数
        value1 += value1_change
        value2 += value2_change
        weight1 += weight1_change
        weight2 += weight2_change

    return weight1, weight2, value1, value2


def testing(data, value, weight1, weight2, value1, value2):
    rightCount = 0
    for i in range(len(data)):
        inputset = np.mat(data[i]).astype(np.float64)
        outputset = np.mat(value[i]).astype(np.float64)
        output2 = sigmoid(np.dot(inputset, weight1) - value1)
        output3 = sigmoid(np.dot(output2, weight2) - value2)

        if output3 > 0.5:
            flag = 1
        else:
            flag = 0

        if value[i] == flag:
            rightCount += 1

        print("预测为%d，实际为%d" % (flag, value[i]))

    return rightCount / len(data)


if __name__ == '__main__':
    data, value = loadData('./horseColicTraining.txt')
    weight1, weight2, value1, value2 = initialize(len(data[0]), len(data[0]), 1)
    for i in range(1500):
        weight1, weight2, value1, value2 = training(data, value, weight1, weight2, value1, value2)
    rate = testing(data, value, weight1, weight2, value1, value2)
    print('正确率为%f' % rate)





