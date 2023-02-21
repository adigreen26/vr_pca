import numpy as np


def oja_pca(X, m, eta, w_prev, epoch, rate=1e-5):
    n, d = X.shape
    error = []
    for s in range(epoch):
        w = w_prev.copy()
        for t in range(m):
            i = np.random.randint(n)
            w = w + eta * X[i].dot(w) * (X[i] - X[i].dot(w) * w)
            w = w / np.linalg.norm(w)
        diff = np.linalg.norm(w_prev - w)
        error.append(diff)
        w_prev = w
        if diff < rate:
            return w_prev, error
    return w_prev, error

