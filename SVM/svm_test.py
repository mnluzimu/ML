import numpy as np
import copy
import random


# In real world, you cannot learn how the data was generated. So do not rely on this function when coding your lab.
def generate_data(dim, num):
    x = np.random.normal(0, 10, [num, dim])
    coef = np.random.uniform(-1, 1, [dim, 1])
    pred = np.dot(x, coef)
    pred_n = (pred - np.mean(pred)) / np.sqrt(np.var(pred))
    label = np.sign(pred_n)
    mislabel_value = np.random.uniform(0, 1, num)
    mislabel = 0
    for i in range(num):
        if np.abs(pred_n[i]) < 1 and mislabel_value[i] > 0.9 + 0.1 * np.abs(pred_n[i]):
            label[i] *= -1
            mislabel += 1
    return x, label, mislabel / num


def select_j_rand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alptha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L

    return aj


class SVM1:
    def __init__(self, dim):
        """
        You can add some other parameters, which I think is not necessary
        """
        self.dim = dim
        self.w = None
        self.b = None

    def smo(self, data_mat_In, class_label, C, toler, max_iter):
        data_matrix = np.mat(data_mat_In)
        label_mat = np.mat(class_label).transpose()
        b = 0
        m, n = np.shape(data_matrix)
        alphas = np.mat(np.zeros((m, 1)))
        iter_num = 0
        while iter_num < max_iter:
            alpha_pairs_changed = 0
            for i in range(m):
                fxi = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
                Ei = fxi - float(label_mat[i])
                if (label_mat[i] * Ei < -toler and alphas[i] < C) or (label_mat[i] * Ei > toler and alphas[i] > 0):
                    j = select_j_rand(i, m)
                    fxj = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                    Ej = fxj - float(label_mat[j])
                    alpha_i_old = copy.deepcopy(alphas[i])
                    alpha_j_old = copy.deepcopy(alphas[j])
                    if label_mat[i] != label_mat[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])
                    if L == H:
                        continue
                    eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i,
                                                                                              :].T - data_matrix[j,
                                                                                                     :] * data_matrix[j,
                                                                                                          :].T
                    if eta >= 0:
                        continue
                    alphas[j] -= label_mat[j] * (Ei - Ej) / eta
                    alphas[j] = clip_alptha(alphas[j], H, L)
                    if abs(alphas[j] - alphas[i]) < 0.001:
                        continue
                    alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                    b_1 = b - Ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - \
                          label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                    b_2 = b - Ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - \
                          label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                    if 0 < alphas[i] and C > alphas[i]:
                        b = b_1
                    elif 0 < alphas[j] and C > alphas[j]:
                        b = b_2
                    else:
                        b = (b_1 + b_2) / 2
                    alpha_pairs_changed += 1

            if alpha_pairs_changed == 0:
                iter_num += 1
            else:
                iter_num = 0
            print("迭代次数：%d" % iter_num)

        return b, alphas

    def caluelate_w(self, data_mat, label_mat, alphas):
        alphas = np.array(alphas)
        data_mat = np.array(data_mat)
        label_mat = np.array(label_mat)
        w = np.dot((np.tile(label_mat.reshape(1, -1).T, (1, 5)) * data_mat).T, alphas)
        return w.tolist()

    def fit(self, X, y):
        """
        Fit the coefficients via your methods
        """
        X = X.tolist()
        y = y.T.squeeze().tolist()
        print(X, y)
        print(len(X), len(y))
        self.b, alphas = self.smo(X, y, 10, 0.001, 40)
        self.w = self.caluelate_w(X, y, alphas)

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        result = []
        w = np.array(self.w)
        for line in X:
            result.append(1 if np.array(line).dot(w) + self.b > 0 else -1)
        return result


if __name__ == "__main__":
    dim = 5
    num = 100
    X, y, mislabel_rate = generate_data(dim, num)
    print(X.shape, y.shape)

    X_train = X[:50]
    y_train = y[:50]
    X_test = X[50:]
    y_test = y[50:]
    svm1 = SVM1(5)
    svm1.fit(X_train, y_train)
    y_predict = svm1.predict(X_test)
    y_predict = np.array(y_predict)
    y_test = y_test.flatten()
    precision = np.sum(y_predict == y_test) / len(y_test)
    print(precision)

