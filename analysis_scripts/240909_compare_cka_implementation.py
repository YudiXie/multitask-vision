# %%
import numpy as np
import rsatoolbox

# %%
# CKA implementation from https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py
def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)

# %%
# test single RDM comparison
X = np.random.randn(150, 64)
Y = np.random.randn(150, 200)

print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

# the following gives the same result as the above
# linear CKA is same as the following, without giving the covariance matrix to compare_cosine_cov_weighted
# sigma_k in compare_cosine_cov_weighted, covariance between pattern estimates,
# not sure if this is the covariance in the RDM space (likly) or the neural space (less likely)
x_data = rsatoolbox.data.Dataset(X)
x_rdm = rsatoolbox.rdm.calc_rdm(x_data)
y_data = rsatoolbox.data.Dataset(Y)
y_rdm = rsatoolbox.rdm.calc_rdm(y_data)
print(f'RSA CKA between X and Y: {rsatoolbox.rdm.compare_cosine_cov_weighted(x_rdm, y_rdm)}')
print(f'RSA CKA between X and X: {rsatoolbox.rdm.compare_cosine_cov_weighted(x_rdm, x_rdm)}')

print('this will give different results than the above')
print(f'RSA compare cosine between X and Y: {rsatoolbox.rdm.compare_cosine(x_rdm, y_rdm)}')
print(f'RSA compare cosine between X and X: {rsatoolbox.rdm.compare_cosine(x_rdm, x_rdm)}')

# %%
# test multiple RDM comparison
X = np.random.randn(150, 64, 3)
Y = np.random.randn(150, 200, 4)

CKA_M1 = np.zeros((3, 4))
for i in range(3):
    for j in range(4):
        CKA_M1[i, j] = linear_CKA(X[:, :, i], Y[:, :, j])

CKA_M2 = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        CKA_M2[i, j] = linear_CKA(X[:, :, i], X[:, :, j])
print('linear_CKA')
print(CKA_M1)
print(CKA_M2)

x_rdms_list = []
for i in range(3):
    x_data = rsatoolbox.data.Dataset(X[:, :, i])
    x_rdms_list.append(rsatoolbox.rdm.calc_rdm(x_data))
x_rdms = rsatoolbox.rdm.concat(x_rdms_list)

y_rdms_list = []
for i in range(4):
    y_data = rsatoolbox.data.Dataset(Y[:, :, i])
    y_rdms_list.append(rsatoolbox.rdm.calc_rdm(y_data))
y_rdms = rsatoolbox.rdm.concat(y_rdms_list)

print('RSA toolbox')
print(rsatoolbox.rdm.compare_cosine_cov_weighted(x_rdms, y_rdms))
print(rsatoolbox.rdm.compare_cosine_cov_weighted(x_rdms, x_rdms))

# %%



