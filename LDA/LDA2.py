import numpy as np
from sklearn import datasets

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class MyLDA:
    def __init__(self):
        self.S = None  # covariate matrix
        self.mu = {}  # the means of each class
        self.prior = {}  # the prior density of each class
        self.labels = None
        self.sigmas = {}  # the covariate of each class

    def fit(self, X, y):
        # 获取所有的类别
        X = np.asarray(X)
        y = np.asarray(y)
        labels = np.unique(y)
        self.labels = labels
        n_samples = X.shape[0]
        # print(labels)
        means = []
        for label in labels:
            # 计算每一个类别的样本均值
            tmp = np.mean(X[y == label], axis=0)
            means.append(tmp)
            self.mu[label] = tmp
            self.prior[label] = len(X[y == label]) / n_samples
        # 如果是二分类的话
        if len(labels) == 2:
            mu = (means[0] - means[1])
            mu = mu[:, None]  # 转成列向量
            B = mu @ mu.T
        else:
            total_mu = np.mean(X, axis=0)
            B = np.zeros((X.shape[1], X.shape[1]))
            for i, m in enumerate(means):
                n = X[y == i].shape[0]
                mu_i = m - total_mu
                mu_i = mu_i[:, None]  # 转成列向量
                B += n * np.dot(mu_i, mu_i.T)

            # 计算S矩阵
        S_t = []
        for label, m in enumerate(means):
            S_i = np.zeros((X.shape[1], X.shape[1]))
            for row in X[y == label]:
                t = (row - m)
                t = t[:, None]  # 转成列向量
                S_i += t @ t.T
            S_t.append(S_i)
        S = np.zeros((X.shape[1], X.shape[1]))

        for s in S_t:
            S += s
        self.S = S

        # S^-1B进行特征分解
        S_inv = np.linalg.inv(S)
        S_inv_B = S_inv @ B
        eig_vals, eig_vecs = np.linalg.eig(S_inv_B)

        # 从大到小排序
        ind = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[ind]
        eig_vecs = eig_vecs[:, ind]
        return eig_vecs

    def predict(self, x):
        x = np.asarray(x)
        decisons = {}
        inv = np.linalg.inv(self.S)
        delta = []

        if x.ndim == 1:  # x is only one sample
            for l in self.labels1:
                t = x @ inv @ self.mu[l] - 0.5 * self.mu[l] @ inv @ self.mu[l] + np.log(self.prior[l])
                decisons[l] = t
                delta.append(t)
            inds_max = np.argmax(delta)
            return self.labels[inds_max]

        else:
            ret = []
            for sam in x:
                for l in self.labels:
                    t = sam @ inv @ self.mu[l] - 0.5 * self.mu[l] @ inv @ self.mu[l] + np.log(self.prior[l])
                    delta.append(t)
                inds_max = np.argmax(delta)
                ret.append(self.labels[inds_max])
                delta = []
            return ret

    def get_accuracy(self, X, y):
        return np.sum(self.predict(X) == y) / len(y)


# 构造数据集
def make_data(centers=3, cluster_std=[1.0, 3.0, 2.5], n_samples=150, n_features=2):
    X, y = make_blobs(n_samples, n_features, centers, cluster_std)
    return X, y


if __name__ == "__main__":
    X, y = make_data(2, [1.0, 1.0])
    np.random.seed(12)
    np.random.shuffle(X)
    np.random.seed(12)
    np.random.shuffle(y)

    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    lda = MyLDA()
    eig_vecs = lda.fit(X, y)
    W = eig_vecs[:, :1]

    colors = ['red', 'green', 'blue']
    fig, ax = plt.subplots(figsize=(10, 8))
    for point, pred in zip(X, y):
        # 画出原始数据的散点图
        ax.scatter(point[0], point[1], color=colors[pred], alpha=0.5)
        # 每个数据点在W上的投影
        proj = (np.dot(point, W) * W) / np.dot(W.T, W)

        # 画出所有数据的投影
        ax.scatter(proj[0], proj[1], color=colors[pred], alpha=0.5)

    plt.show()

    # predict
    print(lda.get_accuracy(X_test, y_test))
