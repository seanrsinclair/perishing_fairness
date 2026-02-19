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


problem_list = ['A', 'B']

for setup in problem_list:
    file_name = "uniform_perishing_table_"+str(setup).replace('.','-')

    print(f'Running for: {file_name}')

    df = pd.read_csv('./data/'+file_name+'.csv')
    df = df.drop('Algorithm', axis=1)
    print(df['NumGroups'].unique())
    # df_group = df[df.NumGroups >= (1/4) * max(df.NumGroups)]
    grouped_df = df.groupby(['Order', 'Norm']).agg({'Value': ['mean', 'sem']}).reset_index()
    grouped_df[('Value', 'sem')] *= 1.96


    tmp = pd.pivot_table(grouped_df, index='Order', columns = 'Norm')

    # print(tmp)


    print(print(tmp.to_latex(index=True,
                formatters={"name": str.upper},
                float_format="{:.4f}".format,
        )) )