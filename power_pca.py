import numpy as np


def power_pca(X, epoch, rate=1e-5):
    mu, sigma = 0, 1  # mean and standard deviation
    v_est = np.random.normal(mu, sigma, size=v.shape)
    # power method
    for s in range(epoch):
        v_est = X.dot(v_est)
        v_est_norm = np.linalg.norm(v_est)
        v_est_t = v_est / v_est_norm
        v_est = v_est_t