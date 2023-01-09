# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 15:40:51 2020

@author: yyimi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import seaborn as sns
import pandas as pd

def Normal_Kernel(xi,xj,alpha,beta):
    return alpha*np.exp(-np.dot((xi-xj),(xi-xj))/(2*beta))
#%%
def finite_no_sample_demo():
    d=20
    x = np.linspace(-1,1,d)
    name = ['x{}'.format(i+1)for i in range(20)]
    cov_mat = np.zeros((d,d))
    mean = np.zeros(d)
    for i in range(d):
        for j in range(d):
            cov_mat[i][j] = Normal_Kernel(x[i],x[j],2,0.1)
    plt.figure(figsize=(18,5))
    for i in range(2):
        plt.subplot(1,3,i+1)
        y = np.random.multivariate_normal(mean, cov_mat)
        plt.plot(name,y)
        plt.scatter(name,y)
    plt.subplot(1,3,3)
    sns.heatmap(cov_mat)
    plt.suptitle('20-Dimension Gaussian')
    plt.show()
#%%
def finite_dimension_sample_demo():
    name = [i+1 for i in range(20)]
    x = np.linspace(-1,1,20)
    
    fixed_x = x[:2]
    fixed_y = np.random.normal(0,0.5,2)
    cov = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            cov[i][j] = Normal_Kernel(fixed_x[i],fixed_x[j],2,0.5)
    K_newf = np.zeros((20,2))
    for i in range(20):
        for j in range(2):
            K_newf[i][j] = Normal_Kernel(x[i],fixed_x[j],2,0.5)
    K_newnew = np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            K_newnew[i][j] = Normal_Kernel(x[i],x[j],2,0.5)
            
    inv = np.linalg.pinv(cov)
    K = K_newnew-K_newf@inv@K_newf.T
    new_mean = K_newf@inv@fixed_y
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)     
    for i in range(4):
        y = np.random.multivariate_normal(new_mean, K)
        plt.plot(name,y,linewidth=1)
        plt.scatter(name,y,linewidth=3)
    plt.scatter([1,2],fixed_y,linewidth=3,color='black',alpha=1,label='fixed dimensions')
    plt.legend()
    
    cov_mat = np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            cov_mat[i][j] = Normal_Kernel(x[i],x[j],2,0.1)
    cov_mat = pd.DataFrame(cov_mat)
    cov_mat.index = name
    cov_mat.columns = name
    plt.subplot(1,2,2)
    sns.heatmap(cov_mat)
    plt.show()
    
    #error bar
    plt.scatter(name,y)
    for x,y,error in zip(name,y,np.diag(K)):
        upper = y+error
        lower = y-error
        plt.vlines(x, lower, upper, 'b')
    plt.scatter([1,2],fixed_y,color='black',label='fixed_dimension')
    plt.legend()
    plt.show()
    
    
#%%
def Gaussian_prior_demo():
    d = 100
    x = np.linspace(-1,1,d)
    cov_mat = np.zeros((d,d))
    mean = np.zeros(d)
    for i in range(d):
        for j in range(d):
            cov_mat[i][j] = Normal_Kernel(x[i],x[j],2,0.1)
            
    plt.title('Gaussian Prior')
    for i in range(5):
        y = np.random.multivariate_normal(mean, cov_mat)
        plt.plot(x,y)
    plt.axhspan(-2,2,facecolor='b', alpha=0.2)
    plt.show()
    
#%%
def Gaussian_posterior_demo():
    sample_size = 3
    sample_x = np.array([np.random.uniform(-1,1) for i in range(sample_size)])
    sample_y = np.array([np.random.uniform(-2,2) for i in range(sample_size)])
    d = 100
    x = np.linspace(-1,1,d)
    cov = np.zeros((sample_size,sample_size))
    for i in range(sample_size):
        for j in range(sample_size):
            cov[i][j] = Normal_Kernel(sample_x[i],sample_x[j],2,0.1)
    K_newf = np.zeros((d,sample_size))
    for i in range(d):
        for j in range(sample_size):
            K_newf[i][j] = Normal_Kernel(x[i],sample_x[j],2,0.1)
    K_newnew = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            K_newnew[i][j] = Normal_Kernel(x[i],x[j],2,0.1)
    
    inv = np.linalg.pinv(cov)
    K = K_newnew-K_newf@inv@K_newf.T
    new_mean = K_newf@inv@sample_y
    mean_new_y = np.zeros(d)
    for i in range(4):
        y = np.random.multivariate_normal(new_mean, K)
        plt.plot(x,y)
        mean_new_y += y
    plt.plot(x, mean_new_y/4,linewidth=3,color='black')
    plt.scatter(sample_x,sample_y,color='r',alpha=1)
    uncertainty = np.diag(K)
    upper = mean_new_y/4 + uncertainty
    lower = mean_new_y/4 - uncertainty
    plt.fill_between(x,lower,upper,color='blue',alpha=0.25)
    plt.show()
    return K


#%%
def kernel_demo():
    sigma = [0.1,0.5,1,10]
    t=2
    plt.figure(figsize=(15,10))
    for i,s in enumerate(sigma):
        plt.subplot(2,2,i+1)
        x = np.random.uniform(-10,10,100)
        x = np.sort(x)
        y = np.exp(-(1/2*s**2)*np.square(x-t))
        plt.scatter(x,y,label='l={}'.format(s))
        plt.xticks([t,t],['target point'])
        plt.legend(fontsize='large')
    plt.show()
#%%  
def covariance_prior_demo():
    d = 100
    x = np.linspace(-1,1,d)
    cov_mat = np.zeros((d,d))
    mean = np.zeros(d)
    for i in range(d):
        for j in range(d):
            cov_mat[i][j] = Normal_Kernel(x[i],x[j],2,0.1)
    l=[0.1,0.5,1,10]
    plt.figure(figsize=(16,10))
    for i,l in enumerate(l):
        cov_mat = np.zeros((d,d))
        mean = np.zeros(d)
        for h in range(d):
            for p in range(d):
                cov_mat[h][p] = Normal_Kernel(x[h],x[p],2,l)
        plt.subplot(2,2,i+1)
        for j in range(5):
            y = np.random.multivariate_normal(mean, cov_mat)
            plt.plot(x,y,linewidth=2)
        plt.axhspan(-2,2,facecolor='b', alpha=0.2,label='L={}'.format(l))
        plt.legend()
            
    plt.show()
#%%