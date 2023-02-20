import numpy as np
import numpy as np


def preprocess_data(X):
    """
    Preprocesses the data matrix X by centering the data and dividing each
    coordinate by its standard deviation times the square root of the dimension.

    Parameters:
    X (numpy.ndarray): The data matrix, where each row represents an observation
        and each column represents a feature.

    Returns:
    numpy.ndarray: The preprocessed data matrix.
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute the standard deviation of each feature
    std_dev = np.std(X_centered, axis=0)

    # Compute the square root of the dimension
    dim_sqrt = np.sqrt(X.shape[1])

    # Divide each coordinate by the standard deviation times the square root of the dimension
    X_preprocessed = X_centered / (std_dev * dim_sqrt)

    return X_preprocessed


def vr_pca(X, m, eta, epoch, rate=1e-5):
    n, d = X.shape
    w_t = np.random.rand(d) - 0.5  # centers the values around zero
    w_t = w_t / np.linalg.norm(w_t)
    error = []
    for s in range(epoch):
        u_t = X.T.dot(X.dot(w_t)) / n
        w = w_t
        for t in range(m):
            i = np.random.randint(n)
            _w = w + eta * (X[i] * (X[i].T.dot(w) - X[i].T.dot(w_t)) + u_t)
            _w = _w / np.linalg.norm(_w)
            w = _w
        d = np.linalg.norm(w_t - w)
        error.append(d)
        w_t = w
        if d < rate:
            return w_t, error
    return w_t, error
