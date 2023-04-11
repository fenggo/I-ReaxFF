import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

def f(x):
    y = np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)
    return y

# X_train = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
X_train = np.array([-2, 0, 2, 4, 6, 8]).reshape(-1, 1)
y_train = f(X_train)
X_test = np.linspace(-2, 10, 10000).reshape(-1, 1)

kernel = ConstantKernel(constant_value=1,constant_value_bounds=(1e-5, 1e5)) + \
                        RBF(length_scale=1, length_scale_bounds=(1e-5, 1e5))
# kernel = ConstantKernel(constant_value=1,constant_value_bounds="fixed") * \
#                         RBF(length_scale=1, length_scale_bounds="fixed")
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gpr.fit(X_train, y_train)
mu, cov = gpr.predict(X_test, return_cov=True)
y_test = mu.ravel()
uncertainty = 1.96 * np.sqrt(np.diag(cov)) # 95%置信区间

plt.figure()
plt.title("l=%.1f sigma_f=%.1f" % (gpr.kernel_.k2.length_scale, gpr.kernel_.k1.constant_value))
plt.fill_between(X_test.ravel(), y_test + uncertainty, y_test - uncertainty, alpha=0.1)
plt.plot(X_test, f(X_test), c="y", label="true")
plt.plot(X_test, y_test, label="predict")
plt.scatter(X_train, y_train, label="train", c="red", marker="D")
plt.legend()

