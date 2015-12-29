import numpy as np
import pylab as pl
import scipy.linalg
import numpy.linalg
import sklearn.decomposition
from sklearn.linear_model import RidgeCV
import random


class LDS:

    def __init__(self, dim):
        self.dim = dim
        self.Q = []
        self.R = []

    def _error(self, data1, data2):
        e = (data1 - data2) ** 2
        e_mean = np.mean(e, axis=0)
        e_mean2 = np.mean(e_mean)
        return e_mean2

    def _inner_predict(self, X):
        DIM = self.dim
        X_pred = X.copy()
        for t in range(2, X.shape[0]):
            x1 = X[t-1].reshape((DIM, 1))
            x2 = X[t-2].reshape((DIM, 1))
            temp1 = np.dot(self.A1, x1)
            temp2 = np.dot(self.A2, x2)
            b = []
            for i in range(self.B.shape[0]):
                b.append(random.gauss(0, B[i, 0]))
            b = np.array(b).reshape((DIM, 1))
            _x = temp1 + temp2 + self.D + b*100
            X_pred[t] = _x.reshape((1, DIM))

        return X_pred

    def predict(self, data, noise=False):
        DIM = self.dim
        L = data.shape[0]

        X = self.pca.transform(data)

        X1 = X[1:-1].T
        X2 = X[0:-2].T

        X_pred = X.copy()

        temp1 = np.dot(self.A1, X1)
        temp2 = np.dot(self.A2, X2)

        _X = temp1 + temp2 + self.D

        X_pred[2:] = _X.T

        Y = self.pca.inverse_transform(X_pred)

        return Y

    def predict_old(self, data, noise=False):
        DIM = self.dim

        X = self.pca.transform(data)

        X_pred = X.copy()
        for t in range(2, X.shape[0]):
            x1 = X[t-1].reshape((DIM, 1))
            x2 = X[t-2].reshape((DIM, 1))
            temp1 = np.dot(self.A1, x1)
            temp2 = np.dot(self.A2, x2)
            if noise:
                b = []
                for i in range(self.B.shape[0]):
                   b.append(random.gauss(0, self.B[i, 0]))
                b = np.array(b).reshape((DIM, 1))
                _x = temp1 + temp2 + self.D + b
            else:
                _x = temp1 + temp2 + self.D
            X_pred[t] = _x.reshape((1, DIM))

        Y = self.pca.inverse_transform(X_pred)

        return Y

    def predict_close(self, y1, y2, L, noise=False):
        DIM = self.dim

        x1 = self.pca.transform(y1)
        x2 = self.pca.transform(y2)

        X_pred = np.zeros((L, DIM))
        X_pred[0] = x1
        X_pred[1] = x2

        for t in range(2, L):
            x1 = X_pred[t-1].reshape((DIM, 1))
            x2 = X_pred[t-2].reshape((DIM, 1))
            temp1 = np.dot(self.A1, x1)
            temp2 = np.dot(self.A2, x2)
            if noise:
                b = []
                for i in range(self.B.shape[0]):
                    b.append(random.gauss(0, self.B[i, 0]))
                b = np.array(b).reshape((DIM, 1))
                _x = temp1 + temp2 + self.D + b
            else:
                _x = temp1 + temp2 + self.D
            X_pred[t] = _x.reshape((1, DIM))

        Y = self.pca.inverse_transform(X_pred)

        return Y

    def fit(self, dataList):

        DIM = self.dim

        data = dataList[0].copy()

        for i in range(1, len(dataList)):
            data = np.vstack([data, dataList[i]])

        pca = sklearn.decomposition.PCA(DIM)
        pca.fit(data)

        self.pca = pca

        train = np.zeros((0, DIM*2+1))
        target = np.zeros((0, DIM))

        for data in dataList:

            X = pca.transform(data)

            temp = np.hstack([X[:-2], X[1:-1]])
            ones = np.ones(X.shape[0]-2)
            temp = np.hstack([ones.reshape((X.shape[0]-2, 1)), temp])
            train = np.vstack([train, temp])
            target = np.vstack([target, X[2:]])

        en = RidgeCV(fit_intercept=True)
        en.fit(train, target)
        pred = map(en.predict, train)
        pred = np.array(pred)
        pred = pred.reshape((pred.shape[0], pred.shape[2]))

        W = en.coef_
        self.D = W[:, 0].reshape((DIM, 1))
        self.A1 = W[:, 1:DIM+1]
        self.A2 = W[:, DIM+1:]

        E = (target - pred) ** 2
        self.B = np.mean(E, axis=0).reshape((DIM, 1))

        # pred1 = self.predict(data)
        # pred2 = self.predict_close(data[0], data[1], data.shape[0])
        #
        # print "[LDS] Error (pred1) =", self._error(data, pred1)
        # print "[LDS} Error (pred2) =", self._error(data, pred2)

        # pl.figure()
        # pl.subplot(3, 1, 1)
        # pl.plot(data)
        # pl.subplot(3, 1, 2)
        # pl.plot(pred1)
        # pl.subplot(3, 1, 3)
        # pl.plot(pred2)
        # pl.show()
