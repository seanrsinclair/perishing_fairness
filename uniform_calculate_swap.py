import warnings;
warnings.filterwarnings('ignore');

from time import sleep
from tqdm.auto import tqdm

import sys
import importlib
import numpy as np
import nbformat
# import plotly.express
# import plotly.express as px
import pandas as pd
import cvxpy as cp
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import seaborn as sns
import helper
import algorithms
from itertools import permutations


num_iterations = 10000

mean_size = 2
var_size = 0.1

FILTER_OE = False
DEBUG = False
EPS = 1








# PAPER DISTRIBUTIONS: ['B', 'c']
# A = Instance 1
# B = Instance 2
# problem_list = ['A', 'B', 'C']
# problem_list = ['A']
# problem_list = ['A','B']
# problem_list = ['A', 'B']
problem_list = ['B']
# problem_list = ['C']
# order_list = ['mean', 'cv', 'random', 'reverse']
# order_list = ['mean', 'cv', 'lcb']
order_list = ['lcb', 'mean', 'cv']
# order_list = ['mean']
# order_list = ['opt']
# order_list = ['mean', 'cv', 'lcb', 'opt']
# order_list = ['lcb']
# order_list = ['opt']
# order_list = ['lcb']
# num_groups = [50]

# num_groups = np.logspace(4, 11, base=1.5, num=10).astype(int)
# num_groups = [50]
num_groups = [50]


def demand_dist(n, mean_size, var_size=.1):
    size = np.maximum(0, np.random.normal(loc=mean_size, scale=np.sqrt(var_size), size=n))
    return size



for setup in problem_list: # useless for loop at the moment, in case we want to run for different alpha variables

    file_name = "uniform_perishing_swap_table_"+str(setup).replace('.','-')
    print(f'Running for: {file_name}')

    data = []

    for n in num_groups:
        n = int(n)


        max_budget = mean_size*n
        
        print(f'Time Horizon: {num_groups}, Number of Resources: {max_budget}')


        if setup == 'A':
            def perish_dist(b, n):
                if b < (max_budget / 2):
                    low_range = n/2 - (b)/2
                    up_range = n/2 + (b)/2
                else:
                    low_range = n
                    up_range = n
                # print(mean, stdev)
                val = np.ceil(np.maximum(0, np.minimum(n, np.random.uniform(low_range, up_range+1))))
                return val
            low_range = np.asarray([n/2 - (b)/2 if b < (max_budget / 2) else n for b in range(max_budget)])
            up_range = np.asarray([n/2 + (b)/2 if b < (max_budget / 2) else n for b in range(max_budget)])
            print(f'Low range: {low_range}')
            print(f'Up range: {up_range}')
            mean_list = (1/2) * (low_range + up_range)
            stdev_list = np.sqrt((1/12)*((up_range - low_range) ** 2))

            if DEBUG: print(f'Mean list: {mean_list}')
            if DEBUG: print(f"STDev: {stdev_list}")
            CV = stdev_list / mean_list
            if DEBUG: print(f"CV: {CV}")


        elif setup == 'B':
            def perish_dist(b, n):
                if b < (mean_size*n / 2):
                    low_range = b+1
                    up_range = b+1
                else:
                    low_range = b+1
                    up_range = n
                # print(mean, stdev)
                # val = np.ceil(np.maximum(0, np.minimum(n, np.random.uniform(low_range, up_range))))
                val = np.random.uniform(low_range, up_range)
                return val
            low_range = np.asarray([b+1 if b < (mean_size*n / 2) else b+1 for b in range(max_budget)])
            up_range = np.asarray([b+1 if b < (mean_size*n / 2) else n for b in range(max_budget)])
            mean_list = (1/2) * (low_range + up_range)
            stdev_list = np.sqrt((1/12)*((up_range - low_range) ** 2))
            if DEBUG: print(f'Mean list: {mean_list}')
            if DEBUG: print(f"STDev: {stdev_list}")
            CV = stdev_list / mean_list
            if DEBUG: print(f"CV: {CV}")


        elif setup == 'C':
            def perish_dist(b, n):
                if b < (mean_size*n / 2):
                    low_range = b+1
                    up_range = b+1
                else:
                    low_range = (b+1 - (mean_size * n / 2))
                    up_range = (b+1 + (mean_size * n / 2))
                    mean = b+1
                # print(mean, stdev)
                val = np.ceil(np.maximum(0, np.minimum(n, np.random.uniform(low_range, up_range))))
                return val
            
            low_range = np.asarray([b+1 if b < (mean_size*n / 2) else (b+1 - (mean_size * n / 2)) for b in range(max_budget)])
            up_range = np.asarray([b+1 if b < (mean_size*n / 2) else (b+1 + (mean_size * n / 2)) for b in range(max_budget)])
            mean_list = (1/2) * (low_range + up_range)
            stdev_list = np.sqrt((1/12)*((up_range - low_range) ** 2))
            if DEBUG: print(f'Mean list: {mean_list}')
            if DEBUG: print(f"STDev: {stdev_list}")
            CV = stdev_list / mean_list
            if DEBUG: print(f"CV: {CV}")

        elif setup == 'D':

            def perish_dist(b, n):
                if b < (mean_size*n / 2):
                    low_range = b+1
                    up_range = b+1
                else:
                    low_range = (b+1 - (mean_size * n / 2))
                    up_range = n
                val = np.ceil(np.maximum(0, np.minimum(n, np.random.uniform(low_range, up_range))))
                return val

            low_range = np.asarray([b+1 if b < (mean_size*n / 2) else (b+1 - (mean_size * n / 2)) for b in range(max_budget)])
            up_range = np.asarray([b+1 if b < (mean_size*n / 2) else n for b in range(max_budget)])
            mean_list = (1/2) * (low_range + up_range)
            stdev_list = np.sqrt((1/12)*((up_range - low_range) ** 2))
            if DEBUG: print(f'Mean list: {mean_list}')
            if DEBUG: print(f"STDev: {stdev_list}")
            CV = stdev_list / mean_list
            if DEBUG: print(f"CV: {CV}")





        offset_prob = helper.check_offset_expiry(perish_dist, lambda n: demand_dist(n, mean_size, var_size), n, max_budget)
        print(f' Probability process is offset expiring: {100*offset_prob}')

        # CALCULATES
        n_upper = helper.n_upper(lambda n: demand_dist(n, mean_size, var_size), n)
        
        num_valid = 0    
        # print(max_budget)
        x_lower_no_perish = (max_budget / n_upper[0])
        print(f'X_lower_no_perishing: {x_lower_no_perish}')



        for _ in tqdm(range(num_iterations)):
            demands = demand_dist(n, mean_size, var_size)
            resource_perish = np.asarray([perish_dist(b,n) for b in range(max_budget)])

            for alloc_order in order_list:


                if alloc_order == 'mean':
                    order = np.lexsort((np.random.random(mean_list.size), mean_list))
                elif alloc_order == 'cv':
                    # order = np.lexsort((np.random.random(mean_list.size),(-1)*CV))
                    order = np.lexsort((np.random.random(mean_list.size), mean_list, -CV))
                elif alloc_order == 'lcb':
                    # lcb = np.maximum(mean_list - 1.96 * stdev_list, low_range)
                    lcb = low_range + .05*(up_range - low_range)
                    # print(f' LCB Values: {lcb}')
                    order = np.lexsort((np.random.random(mean_list.size),lcb))
                    # print(lcb)

                elif alloc_order == 'ucb':
                    # ucb = np.minimum(mean_list + 1.96 * stdev_list, up_range)
                    ucb = low_range + .05*(up_range - low_range)
                    order = np.lexsort((np.random.random(mean_list.size),ucb))

                elif alloc_order == 'random':
                    order = np.random.permutation(max_budget)
                elif alloc_order == 'flipped':
                    order = np.lexsort((np.random.random(mean_list.size),(-1)*mean_list))

                if DEBUG:
                    print(f'Order: {alloc_order}')
                    print(f'Actual Order: {order}')
                    print(f'Perishing Times: {resource_perish}')
                # Calculates number of swaps: i.e. number of times such that resource_perish[b] < 
                swap = helper.average_inversion_given_sample(resource_perish, order)

                if DEBUG: print(f'Number of swaps: {swap}')

                data_dict = {'NumGroups': n, 'Order': alloc_order, 'Norm': 'Swap', 'Value': swap}
                data.append(data_dict)


    df = pd.DataFrame.from_records(data)
    df.to_csv('./data/'+file_name+'.csv', index=False)