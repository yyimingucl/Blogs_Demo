# Author: Yiming Yang
# Contact: zcahyy1@ucl.ac.uk
# Created: 29.11.2020
# Description: Demonstration for Blog - Gaussian Process Classification and its Approximate Infence Approaches

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define Laplacian PDF mu=0, lambda=1 (good approximation)
laplacian_pdf = lambda x: 1/2*np.exp(-np.abs(x))  
approx_gaussian_good = norm(1, 0.5)

# Define two-modal distribution (poor approximation)
norm1 = norm(0.5, 2.5)
norm2 = norm(0.5, 2.5)
def mix_gaussian_pdf(x):
    return 0.5*norm1.pdf(x) + 0.5*norm2.pdf(x)

approx_gaussian_poor = norm(0.2, 0.16+0.36)

x = np.linspace(-10, 10, 500)
fig, axes = plt.subplots(1,2, figsize=(10,4))
axes[0].plot(x, mix_gaussian_pdf(x), label='posterior of f*')
axes[0].set_title("Unsafe to use MAP prediction")
axes[0].set_ylim(0, 1)
axes[0].vlines(x=0.5, ymin=0.1, ymax=0.2, colors='red', label="Mode", linewidth=2.5)
axes[0].legend()

axes[1].plot(x, approx_gaussian_good.pdf(x), label='posterior of f*')
axes[1].set_title("Safe to use MAP prediction")
axes[0].set_ylim(0, 1)
axes[1].vlines(x=1, ymin=0.73, ymax=0.83, colors='red', label="Mode", linewidth=2.5)
axes[1].legend()
