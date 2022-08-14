import numpy as np


def loadData(fileName):
    with open(fileName, 'r', encoding='utf-8') as fp:
        lines = [line.strip().split(sep=', ') for line in fp]
        X = []
        Y = []
        diff_labels = []
        for line in lines:
            # print(line)
            if len(line) < 5:
                continue
            # print(diff_labels)
            data = line[:-1]
            label = line[-1]

            if label == '<=50K':
                Y.append([0])
            else:
                Y.append([1])

            if len(diff_labels) == 0:
                for i in range(len(data)):
                    if data[i].isdigit():
                        diff_labels.append({})
                        data[i] = float(data[i])
                    else:
                        diff_labels.append({data[i]: 0})
                        data[i] = 0

            else:
                for i in range(len(data)):
                    if data[i].isdigit():
                        data[i] = float(data[i])
                    else:
                        if data[i] in diff_labels[i]:
                            data[i] = diff_labels[i][data[i]]
                        else:
                            j = len(diff_labels[i])
                            diff_labels[i][data[i]] = j
                            data[i] = j

            X.append(data)

        return np.array(X), np.array(Y)


class LogisticRegression:
    def __init__(self, alpha, epoch):
        self.alpha = alpha
        self.epoch = epoch
        self.w = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, Y):
        n, m = X.shape
        self.w = np.random.normal(size=(m, 1))
        for i in range(self.epoch):
            hw = self.sigmoid(np.dot(X, self.w))
            err = Y - hw
            self.w = self.w + self.alpha * np.dot(X.T, err)
        return self

    def predict(self, X):
        P0 = self.sigmoid(np.squeeze(np.dot(X, self.w))).tolist()
        pred = [1 if i > 0.5 else 0 for i in P0]
        return np.array(pred)

    def score(self, X, Y):
        pred = self.predict(X)
        count = 0
        for i, j in zip(pred.tolist(), Y.tolist()):
            if i == j:
                count += 1
        return count / pred.shape[0]




if __name__ == '__main__':
    set_path = "adult.data"
    X, Y = loadData(set_path)

    alpha = np.ones_like(X[0]) * 0.01
    LR = LogisticRegression(alpha=0.01, epoch=10000)
    LR.fit(X, Y)
    X_test, Y_test = loadData("adult.test")
    print("训练准确率: ", LR.score(X, Y))
    print("测试准确率: ", LR.score(X_test, Y_test))

