import seaborn as sns 
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

ode_model_names = ['Lotka_Volterra', 'Enzymatic_Reaction', 'Lorenz63']

cur_path = os.path.dirname(os.path.realpath(__file__))

for name in ode_model_names:
    # set paths
    diffrax_result_path = cur_path + '/results/{}/Diffrax.txt'.format(name)
    jax_result_path = cur_path + '/results/{}/JAX.txt'.format(name)
    scipy_result_path = cur_path + '/results/{}/Scipy.txt'.format(name)
    sunode_result_path = cur_path + '/results/{}/Sunode.txt'.format(name)
    plot_save_path = cur_path + '/results/{}/result.png'.format(name)

    m = []
    error = []
    # load results
    with open(diffrax_result_path, 'r') as fp:
        diffrax_result = json.load(fp)
        m.append(np.mean(diffrax_result))
        error.append(np.std(diffrax_result))

    with open(jax_result_path, 'r') as fp:
        jax_result = json.load(fp)
        m.append(np.mean(jax_result))
        error.append(np.std(jax_result))

    with open(scipy_result_path, 'r') as fp:
        scipy_result = json.load(fp)
        m.append(np.mean(scipy_result))
        error.append(np.std(scipy_result))

    with open(sunode_result_path, 'r') as fp:
        sunode_result = json.load(fp)
        m.append(np.mean(sunode_result))
        error.append(np.std(sunode_result))

    df = pd.DataFrame(data=np.vstack([diffrax_result, 
                                     jax_result, 
                                     scipy_result, 
                                     sunode_result]).T,
                        columns=['Diffrax', 'JAX', 'PyMC Default', 'SUNODE'])

    ax = sns.barplot(x="variable", y="value", 
                     data=pd.melt(df), errorbar=("pi", 50), 
                     capsize=.2, errcolor="0.1",
                     linewidth=1, edgecolor="0.5")
    ax.set_xlabel('ODE Solver', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Running Time Comparsion - {} Model'.format(name.replace('_', ' ')), fontweight='bold', fontsize=15)
    for i in ax.containers:
        ax.bar_label(i, labels=['%d±%.2f s' %(mu, e) for mu, e in zip(m,error)], 
                     label_type='edge', padding=4, fontsize=12, fontweight='bold')

    plt.savefig(plot_save_path)
    plt.clf()