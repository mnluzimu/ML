import numpy as np


def loaddataset(filename):
    fp = open(filename)

    # 存放数据
    dataset = []

    # 存放标签
    labelset = []
    for i in fp.readlines():
        a = i.strip().split(sep=',')

        # 每个数据行的最后一个是标签
        dataset.append([float(j) for j in a[:len(a) - 1]])
        if a[-1] == 'Iris-setosa':
            labelset.append([1, 0, 0])
        elif a[-1] == 'Iris-versicolor':
            labelset.append([0, 1, 0])
        else:
            labelset.append([0, 0, 1])

    return dataset, labelset


def parameter_initialization(x, y, z):
    # 隐层阈值
    value1 = np.random.randint(-5, 5, (1, y)).astype(np.float64)

    # 输出层阈值
    value2 = np.random.randint(-5, 5, (1, z)).astype(np.float64)

    # 输入层与隐层的连接权重
    weight1 = np.random.randint(-5, 5, (x, y)).astype(np.float64)

    # 隐层与输出层的连接权重
    weight2 = np.random.randint(-5, 5, (y, z)).astype(np.float64)

    return weight1, weight2, value1, value2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def trainning(dataset, labelset, weight1, weight2, value1, value2):
    # x为步长
    x = 0.01
    # print(dataset)
    for i in range(len(dataset) - 1):
        # 输入数据
        # print(dataset[i])
        inputset = np.mat(dataset[i]).astype(np.float64)
        # print(inputset.shape)
        # 数据标签
        outputset = np.mat(labelset[i]).astype(np.float64)
        # 隐层输入
        # print()
        input1 = np.dot(inputset, weight1).astype(np.float64)
        # 隐层输出
        output2 = sigmoid(input1 - value1).astype(np.float64)
        # 输出层输入
        input2 = np.dot(output2, weight2).astype(np.float64)
        # 输出层输出
        output3 = sigmoid(input2 - value2).astype(np.float64)

        # 更新公式由矩阵运算表示
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


def isCorrect(output3, label):
    for i in range(3):
        if label[i] == 1:
            break
    print(output3.T[0])
    print(label[0])
    if output3.T[i] >= output3.T[0] and output3.T[i] >= output3.T[1] and output3.T[i] >= output3.T[2]:
        return True

    return False


def testing(dataset, labelset, weight1, weight2, value1, value2):
    # 记录预测正确的个数
    # print(dataset)
    rightcount = 0
    for i in range(len(dataset) - 1):
        # 计算每一个样例通过该神经网路后的预测值
        # print(dataset[i])
        inputset = np.mat(dataset[i]).astype(np.float64)
        outputset = np.mat(labelset[i]).astype(np.float64)
        output2 = sigmoid(np.dot(inputset, weight1) - value1)
        output3 = sigmoid(np.dot(output2, weight2) - value2)
        print("output3", output3)
        print("labelset", labelset[i])

        # 确定其预测标签
        if isCorrect(output3, labelset[i]):
            rightcount += 1
        # 输出预测结果
        print("预测为", output3, "实际为", labelset[i])

    # 返回正确率
    return rightcount / len(dataset)


if __name__ == '__main__':
    dataset, labelset = loaddataset('iris.data')
    print(dataset)
    weight1, weight2, value1, value2 = parameter_initialization(len(dataset[0]), len(dataset[0]), 3)
    for i in range(150):
        weight1, weight2, value1, value2 = trainning(dataset, labelset, weight1, weight2, value1, value2)
    rate = testing(dataset, labelset, weight1, weight2, value1, value2)
    print("正确率为%f" % (rate))
