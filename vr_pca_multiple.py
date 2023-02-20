import numpy as np


def vr_pca_multiple(X, m, eta, k, rate=1e-5):
    n, d = X.shape
    W_t = np.random.rand(d, k) - 0.5
    W_t = np.linalg.qr(W_t)[0]

    for s in range(10):
        U_t = X.T.dot(X.dot(W_t)) / n

        W = W_t

        for t in range(m):
            i = np.random.randint(n)
            _W = W + eta * (X[i].reshape(-1, 1) * (X[i].reshape(1, -1).dot(W) - X[i].reshape(1, -1).dot(W_t)) + U_t - W.dot(W.T).dot(U_t))
            _W = np.linalg.qr(_W)[0]
            W = _W

        d = np.linalg.norm(W_t - W)
        W_t = W

        if d < rate:
            return W_t



