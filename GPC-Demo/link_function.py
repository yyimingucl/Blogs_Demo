import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm

# kernel 
def Normal_Kernel(xi,xj,alpha,beta):
    return alpha*np.exp(-np.dot((xi-xj),(xi-xj))/(2*beta))

# link function
def logistic_function(x):
    return 1/(1+np.exp(-x))

normal_cdf = norm.cdf


# Viusualize 
def link_function_effectiveness_demo():
    # Set up GP prior
    d = 500
    x = np.linspace(-10,10,d)
    cov_mat = np.zeros((d,d))
    mean = np.zeros(d)
    for i in range(d):
        for j in range(d):
            cov_mat[i][j] = Normal_Kernel(x[i],x[j], 5, 5)
    # Sample GP prior
    sampled_prior = np.random.multivariate_normal(mean, cov_mat)

    figures, axes = plt.subplots(2 , 3, figsize=(20,10))
    
    axes[0,0].set_axis_off()

    axes[0,1].plot(logistic_function(x))
    axes[0,1].set_title('Logistic Function', fontsize=16, fontweight='bold')
    axes[0,1].set_xticks([])
    axes[0,1].set_xlabel('input x', fontsize=18)
    axes[0,1].yaxis.set_tick_params(labelsize=18)

    axes[0,2].plot(normal_cdf(x))
    axes[0,2].set_title('Probit Function', fontsize=16, fontweight='bold')
    axes[0,2].set_xticks([])
    axes[0,2].set_xlabel('input x', fontsize=18)
    axes[0,2].yaxis.set_tick_params(labelsize=18)

    axes[1,0].plot(sampled_prior)
    axes[1,0].set_title('Sampled GP prior', fontsize=16, fontweight='bold')
    axes[1,0].set_xticks([])
    axes[1,0].set_xlabel('input x', fontsize=18)
    axes[1,0].yaxis.set_tick_params(labelsize=18)
    axes[1,0].set_position([0.07, 0.33, 0.28, 0.3])

    axes[1,1].plot(logistic_function(sampled_prior))
    axes[1,1].set_title('Squashed by Logistic Function', fontsize=16, fontweight='bold')
    axes[1,1].set_xticks([])
    axes[1,1].set_xlabel('input x', fontsize=18)
    axes[1,1].set_yticks([0,1])
    axes[1,1].yaxis.set_tick_params(labelsize=18)

    axes[1,2].plot(normal_cdf(sampled_prior))
    axes[1,2].set_title('Squashed by Probit Function', fontsize=16, fontweight='bold')
    axes[1,2].set_xticks([])
    axes[1,2].set_xlabel('input x', fontsize=18)
    axes[1,2].set_xticks([])
    axes[1,2].set_yticks([0,1])
    axes[1,2].yaxis.set_tick_params(labelsize=18)


link_function_effectiveness_demo()



