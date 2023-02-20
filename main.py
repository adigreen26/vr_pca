import time
import numpy as np
from vr_pca import *
from oja_pca import *

n = 200000
d = 10000

X = np.random.rand(n, d)
X = X - np.mean(X, axis=0)
m = n
r_h = (X **2).sum() / n
eta = 1 / (r_h * np.sqrt(n))
epoch = 60
start_time = time.time()
w, error = oja_pca(X, m, eta, epoch)
s = time.time() - start_time
v = np.mean(X.T.dot(X).dot(w) / w)

print('VR_PCA: %f' % s)
print(w[:10])
print(v)

start_time = time.time()
u_, w_, _ = np.linalg.svd(X.T.dot(X))
s = time.time() - start_time
print('SVD: %f' %s)
print(u_[:,0][:10])
print(w_[0])

U, S, Vt = np.linalg.svd(X)
leading_eigenvector = Vt[0, :]
leading_eigenvalue = S[0]**2
print(leading_eigenvalue)



