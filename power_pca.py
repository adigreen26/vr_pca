import numpy as np


def power_pca(A, w, epoch, rate=1e-5):
    error = []
    w_copy = w.copy()
    for i in range(epoch):
        # Compute Av
        Av = np.dot(A, w)
        # Compute the norm of Av
        norm = np.linalg.norm(Av)
        # Normalize Av
        w = Av / norm
        d = np.linalg.norm(w - w_copy)
        error.append(d)
        w_copy = w
        if d < rate:
            return w, error
    return w, error
