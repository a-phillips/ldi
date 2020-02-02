#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 21:09:31 2020

@author: aphillips


The below code replicates the portfolio optimization in the textbook
"Managing Investment Portfolios, A Dynamic Process", in Chapter 5 - Asset Allocation.

This version adds in a liability which is modeled the same as "Long-Term Bonds".
We force a short 90% allocation to it.

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco


plt.style.use('fivethirtyeight')
np.random.seed(100)


# Asset class return and correlation info

asset_classes = ['UK Equity', 
                 'Ex-UK Equity', 
                 'Intermediate Bonds', 
                 'Long-Term Bonds', 
                 'International Bonds', 
                 'Real Estate',
                 'Liability']

returns =   np.array([0.100, 0.080, 0.040, 0.045, 0.050, 0.070, 0.045])

vols =      np.array([0.150, 0.120, 0.070, 0.080, 0.090, 0.100, 0.080])

corr_matrix = np.array([[1.00, 0.76, 0.35, 0.50, 0.24, 0.30, 0.50],
                        [0.76, 1.00, 0.04, 0.30, 0.36, 0.25, 0.30],
                        [0.35, 0.04, 1.00, 0.87, 0.62, -0.05, 0.87],
                        [0.50, 0.30, 0.87, 1.00, 0.52, -0.02, 1.00],
                        [0.24, 0.36, 0.62, 0.52, 1.00, 0.20, 0.52],
                        [0.30, 0.25, -0.05, -0.02, 0.20, 1.00, -0.02],
                        [0.50, 0.30, 0.87, 1.00, 0.52, -0.02, 1.00]])

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


# Set up the efficient frontier:
#   Optimizing volatility for each of 100 return points
#   Bounding all asset classes to the range (0, 1)
#   Initializing the optimization with equal weight on all assets

return_targets = np.linspace(min(returns), max(returns), 100) - (0.9*returns[-1])
output = []
num_assets = len(asset_classes)
bound_list = [(0, 1) for _ in range(num_assets)]
bound_list[-1] = (-0.9, -0.9)
bounds = tuple(bound_list)
x0 = (num_assets - 1)*[1.0/(num_assets - 1)]
x0.extend([-0.9])
x0 = np.array(x0)


# Minimize vol for each target return

for target in return_targets:
    # Constraints ensure that portfolio weights sum to 1 and that the return
    # equals the target return
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 0.1},
                   {'type': 'eq', 'fun': lambda x: prt_return(x) - target})

    # Use sco.minimize for the optimization
    result = sco.minimize(fun=prt_vol,
                          x0=x0,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
    output.append(result)


# Collect optimal portfolio data from output
    
opt_vols = [data['fun'] for data in output]
opt_rets = return_targets
opt_sharpes = (opt_rets - rf_rate) / opt_vols
opt_wts = [data['x'] for data in output]


# Create a finction to find corner portfolios, which occurs when new assets are
# used in the optimal portfolio
def find_corner_prts(data):
    
    # Define desired output format using a dictionary to ultimately populate a DataFrame
    df_order = ['Prt Num','Exp Ret', 'Std Dev', 'Sharpe']
    df_order.extend(asset_classes)
    df_dict = {}
    for key in df_order:
        df_dict[key] = []
    
    # Loop through all optimal portfolios, checking for if the set of non-zero-weight
    # asset classes changes compared to the most recent one
    curr_asset_set = set()
    prt_num = 1
    for i, check_prt in enumerate(data):
        check_wts = opt_wts[i]
        check_asset_set = set([asset_classes[idx] for idx in range(num_assets) if round(check_wts[idx],4) != 0])
        
        # When we find a change, update the dictionary with corner portfolio data
        if check_asset_set != curr_asset_set:
            df_dict['Prt Num'].append(prt_num)
            df_dict['Exp Ret'].append(round(opt_rets[i], 4))
            df_dict['Std Dev'].append(round(opt_vols[i], 4))
            df_dict['Sharpe'].append(round(opt_sharpes[i], 4))
            for k in range(num_assets):
                df_dict[asset_classes[k]].append(round(check_wts[k], 4))
            curr_asset_set = check_asset_set
            prt_num += 1
    
    # Create DataFrame and ensure columns are in correct order
    final_df = pd.DataFrame(df_dict)
    final_df = final_df[df_order]
    final_df.set_index('Prt Num', inplace=True)
    return final_df


# Collect our corner portfolios
    
corner_prts = find_corner_prts(output)         


# Plot our efficient frontier of volatility for each target return level
plt.scatter(corner_prts['Std Dev'],
            corner_prts['Exp Ret'],
            color='red')
plt.plot(opt_vols,
         opt_rets,
         linestyle='-',
         color='black',
         label='Efficient Frontier')
plt.title('Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.show()


# Create a fill chart showing how asset class weights evolve over the frontier

wts = np.array(opt_wts)
wts = wts.T

plt.stackplot(return_targets, wts[:-1,:], labels=asset_classes)
plt.legend(loc='upper right',fontsize='small')
plt.show()
