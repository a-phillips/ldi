#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 21:09:31 2020

@author: aphillips
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco



plt.style.use('fivethirtyeight')
np.random.seed(100)

# Asset class return and correlation info

asset_classes = ['US Equity', 
                 'Ex-US Equity', 
                 'US Fixed Income', 
                 'Ex-US Fixed Income', 
                 'Private Equity', 
                 'Real Estate',
                 'Natural Resources',
                 'Hedge Funds']

returns =   np.array([0.123, 0.099, 0.101, 0.099, 0.157, 0.078, 0.153, 0.174])

vols =      np.array([0.154, 0.188, 0.067, 0.059, 0.153, 0.055, 0.103, 0.072])

corr_matrix = np.array([[1.00, 0.71, 0.25, 0.22, 0.18, 0.02, 0.43, 0.68],
                        [0.71, 1.00, 0.12, 0.30, 0.40, 0.34, 0.38, 0.55],
                        [0.25, 0.12, 1.00, 0.74, -0.23, -0.05, 0.09, 0.22],
                        [0.22, 0.30, 0.74, 1.00, 0.13, 0.21, 0.08, 0.19],
                        [0.18, 0.40, -0.23, 0.13, 1.00, 0.32, 0.34, 0.20],
                        [0.02, 0.34, -0.05, 0.21, 0.32, 1.00, -0.46, -0.18],
                        [0.43, 0.38, 0.09, 0.08, 0.34, -0.46, 1.00, 0.46],
                        [0.68, 0.55, 0.22, 0.19, 0.20, -0.18, 0.46, 1.00]])

rf_rate = 0.02

# Print asset class info

sharpes = (returns - rf_rate) / vols
info_df = pd.DataFrame({'Class': asset_classes,
                        'Return': returns,
                        'Volatility': vols,
                        'Sharpe': sharpes})
info_df.set_index('Class', inplace=True)
print(info_df)

# Calculate the covariance matrix

vol_diag = np.diag(vols)
cov_matrix = np.dot(vol_diag, np.dot(corr_matrix, vol_diag))

# Define some useful functions

def prt_return(weights):
    return np.sum(np.dot(returns,weights))

def prt_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Set up the efficient frontier

return_targets = np.linspace(min(returns), max(returns), 100)
output = []
num_assets = len(asset_classes)
bounds = tuple([(0, 1) for _ in range(num_assets)])
x0 = np.array(num_assets*[1.0/num_assets])

# Minimize vol for each target return

for target in return_targets:
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: prt_return(x) - target})

    
    result = sco.minimize(fun=prt_vol,
                          x0=x0,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
    output.append(result)

plt.plot([data['fun'] for data in output],
         return_targets,
         linestyle='-',
         color='black',
         label='Efficient Frontier')
plt.title('Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.show()

wts = np.array([data['x'] for data in output])
wts = wts.T

plt.stackplot(return_targets, wts, labels=asset_classes)
plt.legend(loc='upper right',fontsize='small')
plt.show()

