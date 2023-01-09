import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import bernoulli
from scipy.special import expit as sigmoid

from sklearn.datasets import make_moons
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

X, t = make_moons(200, noise=0.4, random_state=17)
X = X*1.2

rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpc = GaussianProcessClassifier(kernel=rbf)

gpc.fit(X, t)

# Obtain optimized kernel parameters
sklearn_theta_0 = gpc.kernel_.k2.get_params()['length_scale']
sklearn_theta_1 = np.sqrt(gpc.kernel_.k1.get_params()['constant_value'])

grid_x, grid_y = np.mgrid[-4:4:200j, -4:4:200j]
grid = np.stack([grid_x, grid_y], axis=-1)

probs = gpc.predict_proba(grid.reshape(-1,2))[:,1].reshape(200,200)


class_0 = X[t==0]
class_1 = X[t==1]

fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
CS = ax.contourf(grid_x, grid_y, probs, 10, origin='lower', alpha=0.9)
CS2 = ax.contour(CS, levels=CS.levels[::2], colors='black', origin='lower')
cbar = fig.colorbar(CS)
cbar.add_lines(CS2)
ax.clabel(CS2, fmt='%2.1f', colors='black', fontsize=16)
ax.scatter(class_0[:,0],class_0[:,1], marker='_', label='class 0', linewidths=2, color='red')
ax.scatter(class_1[:,0],class_1[:,1], marker='+', label='class 1', linewidths=2, color='blue')
ax.set_axis_off()
plt.legend()
plt.savefig('GPC-Cover.png')
plt.show()



