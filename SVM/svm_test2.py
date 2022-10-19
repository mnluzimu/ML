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


class SVM:

    def __init__(self, dim, X, y, c):
        self._y = y
        self._X = X
        self._c = c
        self._b = 0
        self._gram = X.dot(X.T)
        self._dim = dim
        self._alpha = np.zeros((len(y), 1), dtype=np.float64)
        self._w = np.sum(self._alpha * self._y * X, axis=0)
        self._prediction_cache = (np.sum(self._w * X, axis=1) + self._b)[:, np.newaxis]


    def _pick_first(self, tol):
        con1 = self._alpha > 0
        con2 = self._alpha < self._c
        print(self._y.shape, self._prediction_cache.shape)
        err1 = self._y * self._prediction_cache - 1
        err2 = err1.copy()
        err3 = err1.copy()
        err1[(con1 & (err1 <= 0)) | (~con1 & (err1 > 0))] = 0
        err2[((~con1 | ~con2) & (err2 != 0)) | ((con1 & con2) & (err2 == 0))] = 0
        err3[(con2 & (err3 >= 0)) | (~con2 & (err3 < 0))] = 0
        print(err1, err2, err3)
        err = err1 ** 2 + err2 ** 2 + err3 ** 2
        print(err)
        idx = np.argmax(err)
        if err[idx] < tol:
            return
        return idx

    def _pick_second(self, idx1):
        idx = np.random.randint(len(self._y))
        while idx == idx1:
            idx = np.random.randint(len(self._y))
        return idx

    def _get_lower_bound(self, idx1, idx2):
        if self._y[idx1] != self._y[idx2]:
            return max(0., self._alpha[idx2] - self._alpha[idx1])
        return max(0., self._alpha[idx2] + self._alpha[idx1] - self._c)

    def _get_upper_bound(self, idx1, idx2):
        if self._y[idx1] != self._y[idx2]:
            return min(self._c, self._c + self._alpha[idx2] - self._alpha[idx1])
        return min(self._c, self._alpha[idx2] + self._alpha[idx1])

    def _update_alpha(self, idx1, idx2):
        l, h = self._get_lower_bound(idx1, idx2), self._get_upper_bound(idx1, idx2)
        y1, y2 = self._y[idx1], self._y[idx2]
        e1 = self._prediction_cache[idx1] - self._y[idx1]
        e2 = self._prediction_cache[idx2] - self._y[idx2]
        eta = self._gram[idx1][idx1] + self._gram[idx2][idx2] - 2 * self._gram[idx1][idx2]
        a2_new = self._alpha[idx2] + (y2 * (e1 - e2)) / eta
        if a2_new > h:
            a2_new = h
        elif a2_new < l:
            a2_new = l
        a1_old, a2_old = self._alpha[idx1], self._alpha[idx2]
        da2 = a2_new - a2_old
        da1 = -y1 * y2 * da2
        self._alpha[idx1] += da1
        self._alpha[idx2] = a2_new
        self._update_dw_cache(idx1, idx2, da1, da2, y1, y2)
        self._update_db_cache(idx1, idx2, da1, da2, y1, y2, e1, e2)
        self._update_pred_cache(idx1, idx2)

    def _update_dw_cache(self, idx1, idx2, da1, da2, y1, y2):
        self._dw_cache = np.array([da1 * y1, da2 * y2])
        self._w[idx1] += self._dw_cache[0]
        self._w[idx2] += self._dw_cache[1]

    def _update_db_cache(self, idx1, idx2, da1, da2, y1, y2, e1, e2):
        gram_12 = self._gram[idx1][idx2]
        b1 = -e1 - y1 * self._gram[idx1][idx1] * da1 - y2 * gram_12 * da2
        b2 = -e2 - y1 * gram_12 * da1 - y2 * self._gram[idx2][idx2] * da2
        self._db_cache = (b1 + b2) * 0.5
        self._b += self._db_cache

    def _update_pred_cache(self, idx1, idx2):
        self._prediction_cache[idx1][0] = self._w.dot(self._X[idx1]) + self._b
        self._prediction_cache[idx2][0] = self._w.dot(self._X[idx2]) + self._b

    def _prepare(self, **kwargs):
        self._c = kwargs.get("c", self._params["c"])

    def fit(self, tol):
        """
        Fit the coefficients via your methods
        """
        idx1 = self._pick_first(tol)
        if idx1 is None:
            return True
        idx2 = self._pick_second(idx1)
        self._update_alpha(idx1, idx2)

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        result = self._w.dot(X) + self._b
        return result


if __name__ == "__main__":
    dim = 5
    num = 100
    X, y, mislabel_rate = generate_data(dim, num)
    svm = SVM(5, X, y, 0.5)
    svm.fit(0.001)
    X1, y1, mislabel_rate1 = generate_data(dim, num)
    y_predict = svm.predict(X1)
    precision = np.sum(y_predict == y1) / len(y1)
    print(precision)
