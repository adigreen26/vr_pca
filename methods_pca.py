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


def error_calc(X, w, ground_truth):
    # according to the article error
    norm_XTw = np.linalg.norm(X.T.dot(w), ord=2) ** 2  # compute norm of X.T.dot(w)
    return np.log10((1 - norm_XTw / ground_truth))


def vr_pca(X, m, eta, w_t, epoch, ground_truth, rate=1e-6):
    d, n = X.shape
    error = []
    for s in range(epoch):
        u_t = X.dot(X.T.dot(w_t)) / n
        w = w_t
        for t in range(m):
            i = np.random.randint(n)
            _w = w + eta * (X[:, i] * (X[:, i].T.dot(w) - X[:, i].T.dot(w_t)) + u_t)
            _w = _w / np.linalg.norm(_w)
            w = _w
        diff = np.linalg.norm(w_t - w)
        error.append(error_calc(X, w, ground_truth))  # Compute the error and append to the list of errors
        w_t = w
        if diff < rate:
            return w_t, error
    return w_t, error


def power_pca(X, w, epoch, ground_truth, rate=1e-6):
    error = []
    w_copy = w.copy()
    d, n = X.shape
    A = (1 / n) * X.dot(X.T)
    for i in range(epoch):
        # Compute Av
        Av = np.dot(A, w)
        # Compute the norm of Av
        norm = np.linalg.norm(Av)
        # Normalize Av
        w = Av / norm
        diff = np.linalg.norm(w - w_copy)
        error.append(error_calc(X, w, ground_truth))  # Compute the error and append to the list of errors
        w_copy = w
        if diff < rate:
            return w, error
    return w, error


def oja_pca(X, m, eta, w_prev, epoch, ground_truth, rate=1e-6):
    d, n = X.shape
    error = []
    for s in range(epoch):
        w = w_prev.copy()
        for t in range(m):
            i = np.random.randint(n)
            w = w + eta * X[:, i].dot(w) * (X[:, i] - X[:, i].dot(w) * w)
            w = w / np.linalg.norm(w)
        diff = np.linalg.norm(w_prev - w)
        error.append(error_calc(X, w, ground_truth))  # Compute the error and append to the list of errors
        w_prev = w
        if diff < rate:
            return w_prev, error
    return w_prev, error
