import numpy as np


def oja_pca(X, m, eta, w_prev, epoch, rate=1e-5):
    n, d = X.shape
    error = []
    for s in range(epoch):
        w = w_prev.copy()
        for t in range(m):
            i = np.random.randint(n)
            x_i = X[i]
            y_i = x_i.dot(w)
            w = w + eta * y_i * (x_i - y_i * w)
            w = w / np.linalg.norm(w)
        d = np.linalg.norm(w_prev - w)
        error.append(d)
        w_prev = w
        if d < rate:
            return w_prev, error
    return w_prev, error
