import numpy as np


def power_pca(A, v, m):
    for i in range(m):
        # Compute Av
        Av = np.dot(A, v)

        # Compute the norm of Av
        norm = np.linalg.norm(Av)

        # Normalize Av
        v = Av / norm

    return v