# Author: Yiming Yang
# Contact: zcahyy1@ucl.ac.uk
# Created: 29.11.2020
# Description: Demonstration for Blog - Gaussian Process Classification and its Approximate Infence Approaches

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define Laplacian PDF mu=0, lambda=1 (good approximation)
laplacian_pdf = lambda x: 1/2*np.exp(-np.abs(x))  
approx_gaussian_good = norm(0, 1)

# Define two-modal distribution (poor approximation)
norm1 = norm(2, 1)
norm2 = norm(-1, 1)
def mix_gaussian_pdf(x):
    return 0.4*norm1.pdf(x) + 0.6*norm2.pdf(x)

approx_gaussian_poor = norm(0.2, 0.16+0.36)

x = np.linspace(-6, 6, 500)
fig, axes = plt.subplots(1,2, figsize=(10,4))
axes[0].plot(x, mix_gaussian_pdf(x), label='Multi-Modal p')
axes[0].plot(x, approx_gaussian_poor.pdf(x), label='Approximation q')
axes[0].set_title('Poor Approximation', fontsize=18, fontweight='bold')
axes[0].legend()

axes[1].plot(x, laplacian_pdf(x), label='Single-Modal p')
axes[1].plot(x, approx_gaussian_good.pdf(x), label='Approximation q')
axes[1].set_title('Good Approximation', fontsize=18, fontweight='bold')
axes[1].legend()

