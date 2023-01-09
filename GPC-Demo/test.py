import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 

# from scipy.optimize import minimize
# from scipy.stats import bernoulli
# from scipy.special import expit as sigmoid

from sklearn.datasets import make_moons

sns.set_theme(style="darkgrid")

# make dataset 
X, t = make_moons(400, noise=0.25)

data = np.concatenate((X,t[None,:].T), axis=1)
data = pd.DataFrame(data, columns=["x1","x2","label"])

sns.scatterplot(data, x="x1", y="x2", hue="label")
plt.title('2D training dataset')
plt.legend()


# plt.scatter(X[0], X[1], )
