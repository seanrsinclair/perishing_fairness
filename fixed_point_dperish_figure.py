import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

from matplotlib.backends.backend_pgf import FigureCanvasPgf
import matplotlib



matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)



plt.style.use('PaperDoubleFig.mplstyle.txt')
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
plt.rcParams['text.usetex'] = True



# Helper functions to calculate \tau_b(x: \sigma)

# Three resources, a b c
# \sigma(a) = 2
# \sigma(b) = 3
# sigma(c) = 1

sigma_list = {'a': 2, 'b': 3, 'c': 1}
N_list = np.asarray([1,3,4])


def tau(X, b):
    threshold = sigma_list[b]
    for t in range(N_list.shape[0]):
        allocated = N_list[t] * X
        if allocated >= threshold:
            return t + 1  # since t starts from 0 but time starts from 1
    return float('inf')  # if condition is never met



# print(tau(3/4, 'a'))
# print(tau(3/4, 'b'))
# print(tau(3/4, 'c'))


distr_list = {
     'a': {1 : 1/2, 3 : 1/2},
     'b': {2 : 1/2, 4: 1/2},
     'c': {1: 1/2, 2: 1/2}
     }

def mu(x, T=np.inf):
    mu_tot = 0
    for b in sigma_list:
        tau_b = tau(x, b)
        cutoff = min(T, tau_b)
        prob = sum(p for t, p in distr_list[b].items() if t < cutoff)
        mu_tot += prob
    return mu_tot




x_values = np.arange(0.01, 1, 0.001) # avoid decision by zero by starting from a small value

y_values_list = [(len(distr_list) - mu(x, 3)) / 4 for x in x_values]
y_values_list_two = [x for x in x_values]


# Find the largest x such that x <= B - Δ(x)/N
valid_indices = [i for i, (x, y) in enumerate(zip(x_values, y_values_list)) if x <= y]
if valid_indices:
    max_valid_index = valid_indices[-1]
    x_star = x_values[max_valid_index]
    y_star = y_values_list[max_valid_index]
else:
    x_star = None

print(f'x_star: {x_star}, y_star: {y_star}')


plt.figure(figsize=(10, 6))



# # Step plot for solid curve (right-continuous steps)
# plt.step(x_values, y_values_list, where='post', label=r'$B - \overline{\Delta}(X)/\overline{N}$', color='black')

legend_flag = False

# Plot step function manually without vertical lines
for i in range(len(x_values) - 1):
    x0 = x_values[i]
    x1 = x_values[i + 1]
    y = y_values_list[i]

    if y_values_list[i] != y_values_list[i + 1]:
        # Plot horizontal line segment
        plt.plot([x0, x1], [y, y], color='black')
        # Closed circle at left
        plt.plot(x1, y_values_list[i+1], 'ko', markersize=10)

        # Open circle at right (step end)
        plt.plot(x0, y, marker='o', markersize=10, markerfacecolor='white', markeredgecolor='black')
    else:
        # Continue flat segment (no change)
        if legend_flag == False:
            plt.plot([x0, x1], [y, y], color='black', label=r'$(B - \overline{\Delta}(X))/\overline{N}$')
            legend_flag = True
        else:
            plt.plot([x0, x1], [y, y], color='black', label=None)

# Final closed circle for last point
plt.plot(x_values[-1], y_values_list[-1], 'ko', markersize=10)


plt.plot(x_values, y_values_list_two, color='blue', linestyle='--', label=r'$X$')

plt.scatter(x_star, y_star, marker='*', s=300,color='red', label = r'$\underline{X}$',zorder=10)



plt.xlabel(r'$X$')
plt.legend()
# plt.grid(True)
plt.tight_layout()
# plt.show()

plt.savefig('./figures/fixed_point_example.pdf', bbox_inches = 'tight',pad_inches = 0.01, dpi=900)






# # Define the functions
# def func1(x):
#     return (min(3, math.ceil(1/x))-1)/2

# def func2(x):
#     return (min(3, math.ceil(2/x))-1)/3

# def func3(x):
#     return 1/2

# # Generate x values
# x_values = np.arange(0.01, 1.002, 0.001)  # Avoid division by zero, start from a small value

# # Calculate y values for each function
# y_values1 = {x: func1(x) for x in x_values}
# y_values2 = {x: func2(x) for x in x_values}
# y_values3 = {x: func3(x) for x in x_values}
# for x in x_values:
#   if x >= 0.495 and x <= 0.505:
#     y_values1[x] = np.nan
#   if x >= 1:
#     y_values1[x] = np.nan
#     y_values2[x] = np.nan


# print(y_values1)
# # raise ValueError()


# sum_val = {}
# rhs = {}
# for x in x_values:
#   if np.isnan(y_values1[x]):
#     sum_val[x] = np.nan
#     rhs[x] = np.nan
#   else:
#     sum_val[x] = y_values1[x] + y_values2[x] + y_values3[x]
#     rhs[x] = 1-sum_val[x]/3






# # Plot the functions
# plt.figure(figsize=(8, 6))

# plt.plot(x_values, [y_values1[x] for x in x_values], label=r'$b=1$',color='blue')
# plt.scatter(1, 0, marker='o', s=50,color='blue')
# plt.scatter(0.999, 0.5, marker='o', facecolors='none',s=50,color='blue')
# plt.scatter(1, 1.0/3, marker='o', s=50,color='orange')
# plt.scatter(0.999, 2.0/3, marker='o', facecolors='none',s=50,color='orange')
# plt.scatter(0.499, func1(0.499), color='blue', marker='o', facecolors='none', s=50)
# plt.scatter(0.5, func1(0.5), color='blue', marker='o', s=50)
# plt.plot(x_values, [y_values2[x] for x in x_values], label=r'$b=2$',color='orange')
# plt.plot(x_values, [y_values3[x] for x in x_values], label=r'$b=3$',color='green',ls='--')

# plt.title(r'Worst-case perishing probability vs. $X$')
# plt.xlabel(r'$X$')
# plt.ylabel(r'$P(T_b < \min\{T, \tau_b(1 \mid X,\sigma)\})$')
# plt.legend()
# plt.show()

# plt.plot(x_values, list(rhs.values()),color='black',label=r'$\frac{B-\overline{\Delta}(X)}{\overline{N}}$')
# plt.plot(x_values,x_values,color='black',ls='--',label=r'$X$')
# sum_val[0.499] = func1(0.499) + func2(0.499) + func3(0.499)
# rhs[0.499] = 1-sum_val[0.499]/3
# sum_val[0.5] = func1(0.5) + func2(0.5) + func3(0.5)
# rhs[0.5] = 1-sum_val[0.5]/3
# plt.scatter(0.499, rhs[0.499], color='black', marker='o', facecolors='none', s=50)
# plt.scatter(0.5, rhs[0.5], color='black', marker='o', s=50)
# plt.scatter(0.2779999999999998, rhs[0.2779999999999998],marker='*', s=100,color='red')
# plt.scatter(1,0.7222222222222223,marker='o',s=50,color='black')
# plt.scatter(1,0.44444444444444453,marker='o',s=50,facecolors='none',color='black')
# # plt.scatter(0, 0,marker='o', s=50,color='black')
# # plt.scatter(0,0.2777777777777778,marker='o',facecolors='none',s=50,color='black')
# plt.xlabel(r'$X$')
# plt.title(r'Solving $\sup\left\{X \mid X \leq \frac{B-\overline{\Delta}(X)}{\overline{N}}\right\}$')
# plt.legend()
# plt.show()






# def cdf(id,half_value_thresholds,x):
#     min_value = min(4, math.ceil(id/x))
#     # print(id,x,math.ceil(id/x),min_value)
#     if min_value < min(half_value_thresholds):
#       return 0
#     elif min_value in half_value_thresholds:
#       return 1/2
#     else:
#       return 1

# thresh = {1: [2], 2: [2,3,4], 3: [3], 4: [4]}
# plt_dct = {}
# yval = np.arange(0.01,1.002,0.001)
# xval = [round(e,3) for e in yval]
# ids = [1,2,3,4]
# # discontinuities = [0.5]
# # ids = [1,]
# for id in ids:
#   plt_dct[id] = {}
#   for x in xval:
#     plt_dct[id][x] = cdf(id,thresh[id],x)

# agg_dct = {}
# rhs = {}
# discontinuities = {}
# for x in xval:
#   agg_dct[x] = sum(plt_dct[id][x] for id in ids)
#   rhs[x] = 1-agg_dct[x]/len(ids)
#   if x == 0.499 or x == 0.999:
#     discontinuities[x] = rhs[x]
#   if x >= 0.495 and x <= 0.505 and x != 0.5:
#     rhs[x] = np.nan
#   if x == 0.999:
#     rhs[x] = np.nan


# plt.plot(xval,list(rhs.values()),color='black',label=r'$\frac{B-\overline{\Delta}(X)}{\overline{N}}$')
# plt.plot(xval, xval,color='black',ls='--',label=r'$X$')
# plt.scatter(0.5,rhs[0.5],color='black',marker = 'o',s=50)
# plt.scatter(0.5,discontinuities[0.499],color='black',marker = 'o',facecolors='none',s=50)
# plt.scatter(0.25,0.25,color='red',marker='*',s=500)
# plt.scatter(1,rhs[1],color='black',marker = 'o',s=50)
# plt.scatter(1,discontinuities[0.999],color='black',marker = 'o',facecolors='none',s=50)
# plt.xlabel(r'$X$')
# # plt.title(r'Solving $\sup\left\{X \mid X \leq \frac{B-\overline{\Delta}(X)}{\overline{N}}\right\}$')
# plt.legend()
# plt.plot()
# # plt.show()
# plt.savefig('./figures/fixed_point_example.pdf', bbox_inches = 'tight',pad_inches = 0.01, dpi=900)