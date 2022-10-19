import numpy as np
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


# you can do anything necessary about the model

class SVM1:
    def __init__(self, dim, X, y, c):
        """
        You can add some other parameters, which I think is not necessary
        """
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

    def _update_alpha(self, idx1, idx2):
        pass

    def fit(self, X, y):
        """
        Fit the coefficients via your methods
        """
        alpha = np.random.random(len(y))
        idx1 = self._pick_first(tol=0.0001)
        while idx1 is not None:
            idx2 = self._pick_second(idx1=idx1)
            print(idx1, idx2)
            self._update_alpha(idx1=idx1, idx2=idx2)
            idx1 = self._pick_first(tol=0.0001)

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """

if __name__ == "__main__":
    svm =