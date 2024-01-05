from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from methods_pca import *
import time
from tensorflow.keras.datasets import cifar10
import random
random.seed(2610)


# Load the cifar10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Combine the train and test sets into one dataset and center the data
X = np.concatenate([x_train, x_test], axis=0)
# Convert RGB to grayscale
X_train_gray = np.dot(X, [0.2989, 0.5870, 0.1140])
# Flatten images
X = X.reshape(X.shape[0], -1).T
# pre-processing images
X = preprocess_data(X)

# ground truth leading eigenvector v of the matrix
eigenvalues, eigenvectors = eigsh(X.dot(X.T))
v = eigenvectors[:, np.argmax(abs(eigenvalues))]
ground_truth = eigenvalues[np.argmax(abs(eigenvalues))]
print('Leading eigenvector ground truth:', eigenvalues[np.argmax(abs(eigenvalues))])

# parameter initialization
epoch = 20
d, n = X.shape
m = n
r_h = (X ** 2).sum() / n
eta = 1 / (r_h * np.sqrt(n))
# Same starting vector
w_t = np.random.rand(d) - 0.5  # centers the values around zero
w_t = w_t / np.linalg.norm(w_t)

# Apply vr_pca algorithm
start_time = time.time()
w_vr, error_vr = vr_pca(X, m, eta, w_t, epoch, ground_truth)
s = time.time() - start_time
print('VR_PCA: %f sec' % s)
# Compute variance explained by top principal component
v = np.mean(X.dot(X.T).dot(w_vr) / w_vr)
print('Leading eigenvector vr_pca:', v)

# Apply vr_pca hybrid algorithm
start_time = time.time()
initial_guess, _ = oja_pca(X, m, eta, w_t, 1, ground_truth)
w_vr, error_vr_hybrid = vr_pca(X, m, eta, initial_guess, epoch, ground_truth)
s = time.time() - start_time
print('VR_PCA_HYBRID: %f sec' % s)
# Compute variance explained by top principal component
v = np.mean(X.dot(X.T).dot(w_vr) / w_vr)
print('Leading eigenvector vr_pca_hybrid:', v)

# Apply power iteration algorithm
start_time = time.time()
w_power, error_power = power_pca(X, w_t, epoch, ground_truth)
s = time.time() - start_time
print('POWER_PCA: %f sec' % s)
# Compute variance explained by top principal component
v = np.mean(X.dot(X.T).dot(w_power) / w_power)
print('Leading eigenvector power_pca:', v)

# Apply oja_pca algorithm eta = same as all
start_time = time.time()
w_oja, error_oja1 = oja_pca(X, m, eta, w_t, epoch, ground_truth)
s = time.time() - start_time
print('OJA_PCA: %f sec' % s)
# Compute variance explained by top principal component
v = np.mean(X.dot(X.T).dot(w_oja) / w_oja)
print('Leading eigenvector oja_pca:', v)

# Apply oja_pca algorithm eta = eta-0.01
start_time = time.time()
w_oja, error_oja2 = oja_pca(X, m, eta-0.01, w_t, epoch, ground_truth)
s = time.time() - start_time
print('OJA_PCA: %f sec' % s)
# Compute variance explained by top principal component
v = np.mean(X.dot(X.T).dot(w_oja) / w_oja)
print('Leading eigenvector oja_pca:', v)

# Apply oja_pca algorithm eta = eta+0.01
start_time = time.time()
w_oja, error_oja3 = oja_pca(X, m, eta+0.01, w_t, epoch, ground_truth)
s = time.time() - start_time
print('OJA_PCA: %f sec' % s)
# Compute variance explained by top principal component
v = np.mean(X.dot(X.T).dot(w_oja) / w_oja)
print('Leading eigenvector oja_pca:', v)

plt.plot(np.arange(1, len(error_vr_hybrid)+1)*2, error_vr_hybrid, 'b', linewidth=0.5)
plt.plot(np.arange(1, len(error_vr)+1)*2, error_vr, 'b')
plt.plot(np.arange(1, len(error_power)+1), error_power, '--r')
plt.plot(np.arange(1, len(error_oja1)+1), error_oja1, '--g')
plt.plot(np.arange(1, len(error_oja2)+1), error_oja2, color='#00FF00', linestyle='dashed')
plt.plot(np.arange(1, len(error_oja3)+1), error_oja3, color='#BBF90F', linestyle='dashed')
plt.legend(['VR-PCA Hybrid', 'VR-PCA', 'Power iterations', 'Oja, ηt=η', 'Oja, ηt=η-0.01', 'Oja, ηt=η+0.01'])
plt.ylabel('log-error')
plt.xlabel('# data passes')
plt.title('Cifar-10 Dataset')
plt.ylim([-10, 0])
plt.xticks([0, 5, 10, 15, 20])
plt.show()

## Check for different learning rates
labels = []
for etot in [eta/20, eta/10, eta, eta*10, eta*20]:
    w_vr, error_vr = vr_pca(X, m, etot, w_t, epoch, ground_truth)
    if etot == eta:
        labels.append('η =' + str(etot)[:-10] + ' (original η)')
    else:
        labels.append('η =' + str(etot)[:-10])
    plt.plot(np.arange(1, len(error_vr)+1)*2, error_vr)
plt.legend(labels)
plt.ylabel('log-error')
plt.xlabel('# data passes')
plt.title('Cifar-10 Dataset - VR_PCA with different learning rates')
plt.ylim([-10, 0])
plt.xticks([0, 5, 10, 15, 20])
plt.show()