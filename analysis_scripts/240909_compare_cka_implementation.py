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
X = np.random.randn(150, 64)
Y = np.random.randn(150, 200)

print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

# this gives the same result as the above
# linear CKA is same as the following, without giving the covariance matrix to compare_cosine_cov_weighted
x_data = rsatoolbox.data.Dataset(X)
x_rdm = rsatoolbox.rdm.calc_rdm(x_data)
y_data = rsatoolbox.data.Dataset(Y)
y_rdm = rsatoolbox.rdm.calc_rdm(y_data)
print(f'RSA CKA between X and Y: {rsatoolbox.rdm.compare_cosine_cov_weighted(x_rdm, y_rdm)}')
print(f'RSA CKA between X and X: {rsatoolbox.rdm.compare_cosine_cov_weighted(x_rdm, x_rdm)}')

# %%
# this will give different results than the above
print(f'RSA compare cosine between X and Y: {rsatoolbox.rdm.compare_cosine(x_rdm, y_rdm)}')
print(f'RSA compare cosine between X and X: {rsatoolbox.rdm.compare_cosine(x_rdm, x_rdm)}')
