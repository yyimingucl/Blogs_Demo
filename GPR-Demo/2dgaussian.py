# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 13:29:09 2020

@author: yyimi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
def plot_2dgaussian(Miu,Sigma):
    '''
    Parameters
    ----------
    Miu : Mean
    Sigma : Covaraince matrix
    Returns
    -------
    '''
    pho = Sigma[0][1]
    x = np.linspace(-3,3,100)
    y = np.linspace(-3,3,100)
    sample_size = 5
    X,Y = np.meshgrid(x, y)
    gaussian = multivariate_normal(mean=Miu,cov=Sigma)
    Z = np.zeros((100,100))
    for i in range(100):
        Z[i] = gaussian.pdf([(x[i],y[j]) for j in range(100)])
       
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.contour(X,Y,Z)
    plt.xlabel('x1')
    plt.ylabel('x2')
    data = {"x1":[],"x2":[]}
    for s in range(sample_size):
        v1,v2 = np.random.multivariate_normal(Miu, Sigma)
        plt.plot(v1,v2,'bo')
        plt.vlines(v1, -3, v2, colors = "r",linestyles = "dashed")
        plt.hlines(v2, -3, v1, colors = "b",linestyles = "dashed")
        data['x1'].append(v1)
        data['x2'].append(v2)
    
    plt.subplot(1,2,2)
    name = list(data.keys())
    values = list(data.values())
    plt.plot(name,values)
    plt.suptitle('2d Gaussian with covariance {}'.format(pho))
    plt.show()
    

if __name__ == '__main__':
    random_seed = 10
    np.random.seed(random_seed)
    cov = [0,0.7,0.95]
    Miu = np.array([0,0])
    for c in cov:
        Sigma = np.array([[1,c],[c,1]])
        plot_2dgaussian(Miu, Sigma)
    
    
    

