# Online Fair Allocation of Perishable Resources

In this repository we include all of the code used for generating the figures and numerical simulations for the paper [Online Fair Allocation of Perishable Resources](https://arxiv.org/abs/2406.02402).

## Supplementary Files
- `algorithms.py` implements all of the algorithms
- `helper.py` provides utility functions for calculating $\underline{X}$, $\Delta(X)$, etc

## Running a Simulation
- `run_uniform.py` runs the simulations under the non-iid perishing distribution with different allocation schedules
- `run_geometric.py` runs the simulations under (synthetic) Geometric perishing
- `run_geometric_tradeoff.py` runs the simulations under (synthetic) Geometric perishing to generate the trade-off curves
- `run_geometric_real_data.py` runs simulations using the real-world ginger perishing data

## Creating the Figures and Tables
- `uniform_plots.py` creates regret plots across different horizons and policies under the non-iid perishing model  
- `uniform_calculate_swap.py` computes SWAP statistics under different orderings 
- `uniform_data_swap_table.py` creates tables summarizing SWAP statistic across all orderings  
- `uniform_data_table.py` creates tables of average performance for each algorithm  
- `geometric_plots.py` produces main metric plots for the geometric perishing model  
- `geometric_tradeoff_figure.py` creates the fairness-efficiency tradeoff plot  
- `geometric_dperish_figure.py` and `fixed_point_dperish_figure.py` analyze perishing-related behaviors for the main text of the paper  
- `geometric_real_data_table.py` summarizes regret on real data using the geometric model  