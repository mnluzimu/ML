from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import math


def create_data(N, M):
    X = mat(zeros((N, M)))
    Y = mat(zeros((N, M)))
    k = 0

    for i in range(M):
        for j in range(N):
            X[j, i] = random.uniform(2 + k * 8, 8 + k * 8)
            Y[j, i] = random.uniform(2, 15) + 10
        k += 1

    return X,Y


def LDA(x1, x2):
    n1 = shape(x1)[0]
    n2 = shape(x2)[0]
    m = n1
    print("m = ", m)
    x1_mean_vector = np.mean(x1.T)
    x2_mean_vector = np.mean(x2.T)
    print("x1_mean_vector = ", x1_mean_vector)
    print("m2_mean_vector = ", x2_mean_vector)
    sigma1 = (x1 - x1_mean_vector).T * (x1 - x1_mean_vector)
    sigma2 = (x2 - x2_mean_vector).T * (x2 - x2_mean_vector)
    Sw = sigma1 + sigma2
    print("Sw = ", Sw)
    w = Sw.I * (x1_mean_vector - x2_mean_vector)
    return w


def show_experiment_plot(X, Y, W):
    n, m = shape(X)
    x_label_data_list = arange(0, 30, 0.1)
    x_label_mat = mat(zeros((1, 300)))

    for i in range(300):
        x_label_mat[0, i] = x_label_data_list[i]
    y_label_mat = W * x_label_mat

    plt.plot(x_label_mat, y_label_mat, "ob")
    for i in range(m):
        if i % 2 == 1:
            plt.plot(X[:, i], Y[:, i], "or")
        else:
            plt.plot(X[:, i], Y[:, i], "og")

    plt.show()


if __name__ == "__main__":
    x1, x2 = create_data(10, 2)
    w = LDA(x1[:, 0], x1[:, 1])
    print("W = ", w)
    show_experiment_plot(x1, x2, w)
    print("------------------------")