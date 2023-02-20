import numpy as np
import matplotlib.pyplot as plt
from vr_pca import *
from oja_pca import *
from power_pca import *
import time
from tensorflow.keras.datasets import fashion_mnist

# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Combine the train and test sets into one dataset and center the data
X = np.concatenate([x_train, x_test], axis=0).T
y = np.concatenate([y_train, y_test], axis=0)
X = X.reshape(-1, X.shape[-1]).T
X = preprocess_data(X)


m = X.shape[0]
r_h = (X ** 2).sum() / X.shape[0]
eta = 1 / (r_h * np.sqrt(X.shape[0]))
epoch = 11
n, d = X.shape
# Same starting vector
w_t = np.random.rand(d) - 0.5  # centers the values around zero
w_t = w_t / np.linalg.norm(w_t)

# Apply power iteration algorithm
start_time = time.time()
w_power= power_pca(X.T.dot(X), w_t, m)
s = time.time() - start_time
print('POWER_PCA: %f sec' % s)
# Compute variance explained by top principal component
v = np.mean(X.T.dot(X).dot(w_power) / w_power)
print('Leading eigenvector power_pca:', v)

# Apply oja_pca algorithm
start_time = time.time()
w_oja, error_oja = oja_pca(X, m, eta, w_t, epoch)
s = time.time() - start_time
print('OJA_PCA: %f sec' % s)
# Compute variance explained by top principal component
v = np.mean(X.T.dot(X).dot(w_oja) / w_oja)
print('Leading eigenvector oja_pca:', v)

# Apply vr_pca algorithm
start_time = time.time()
w_vr, error_vr = vr_pca(X, m, eta, w_t, epoch)
s = time.time() - start_time
print('VR_PCA: %f sec' % s)
# Compute variance explained by top principal component
v = np.mean(X.T.dot(X).dot(w_vr) / w_vr)
print('Leading eigenvector vr_pca:', v)

plt.plot(np.arange(len(error_oja)), np.log10(error_oja))
plt.plot(np.arange(len(error_vr)), np.log10(error_vr))
plt.legend(['oja', 'vr_pca'])
plt.show()


from scipy.sparse.linalg import eigsh
# ground truth leading eigenvector v of the matrix
eigenvalues, eigenvectors = eigsh(X.T.dot(X))
v = eigenvectors[:, np.argmax(abs(eigenvalues))]
print('Leading eigenvector ground truth:', eigenvalues[np.argmax(abs(eigenvalues))])
