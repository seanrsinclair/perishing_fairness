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



sigma_list = {'a': 2, 'b': 3, 'c': 1}
N_list = np.asarray([1,3,4])


def tau(X, b):
    threshold = sigma_list[b]
    for t in range(N_list.shape[0]):
        allocated = N_list[t] * X
        if allocated >= threshold:
            return t + 1  # since t starts from 0 but time starts from 1
    return float('inf')  # if condition is never met


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