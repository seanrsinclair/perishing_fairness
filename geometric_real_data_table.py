import warnings;
warnings.filterwarnings('ignore');

from time import sleep
from tqdm.auto import tqdm

import helper

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


INCLUDE_PUA = True


algo_list = ['static_x_lower', 'static_b_over_n', 'hope_guardrail_35', 'og_hope_guardrail_35']



file_name = "geometric_perishing_2_real_data_"
print(f'Running for: {file_name}')

df = pd.read_csv('./data/'+file_name+'.csv')


df['Log Value'] = np.log(df['Value'])
df['Log NumGroups'] = np.log(df['NumGroups'])




modified_algo_list = algo_list

print(modified_algo_list)

df = df[df['Algorithm'].isin(modified_algo_list)]

algo_list = ['static_x_lower', 'static_b_over_n', 'hope_guardrail_12', 'og_hope_guardrail_12']


df = df.replace({'static_x_lower':r'Static $\underline{X}$',
                    'static_b_over_n':r'Static $B / \overline{N}$',
                    'hope_guardrail_12':'Perish-Guardrail',
                    'og_hope_guardrail_12':'Vanilla-Guardrail'})


df_group = df
df_group = df_group.groupby(['Algorithm', 'Norm'], as_index=False).agg(
                {'Value':['mean', 'sem']})

df_group.loc[:, ('Value', 'sem')] = df_group.loc[:, ('Value', 'sem')] * 1.96


tmp = pd.pivot_table(df_group, index='Algorithm', columns = 'Norm')

print(tmp)


print(print(tmp.to_latex(index=True,
              formatters={"name": str.upper},
              float_format="{:.1f}".format,
    )) )