import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv('./data_analysis/ginger_data.csv')

sale_qty = df['SaleQty'].dropna()

# Fit normal using sample statistics
mu = np.mean(sale_qty)
sigma = np.std(sale_qty)

# Rounded values for legend
mu_r = round(mu, 2)
sigma_r = round(sigma, 2)

# ----------------------------
# Plot style
# ----------------------------
plt.style.use('PaperDoubleFig.mplstyle.txt')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# ----------------------------
# Histogram
# ----------------------------
sns.histplot(
    sale_qty,
    bins=30,
    stat='density',
    color=sns.color_palette("colorblind")[0],
    alpha=0.5,
    edgecolor='black',
    ax=ax
)

# ----------------------------
# Overlay fitted Normal
# ----------------------------
x_vals = np.linspace(
    0,
    max(sale_qty.max(), mu + 4*sigma),
    500
)

ax.plot(
    x_vals,
    norm.pdf(x_vals, mu, sigma),
    linewidth=3,
    linestyle='--',
    label=rf'$N({mu_r},{sigma_r}^2)$'
)

# ----------------------------
# Labels
# ----------------------------
ax.set_xlabel(r'Sale Quantity')
ax.set_ylabel(r'Density')

ax.legend()

fig.savefig(
    './figures/ginger_saleqty_histogram.pdf',
    bbox_inches='tight',
)

# plt.show()


from scipy.stats import shapiro

stat, p_value = shapiro(sale_qty)

print(f"Shapiro-Wilk statistic: {stat:.4f}")
print(f"p-value: {p_value:.4g}")

from scipy.stats import anderson

result = anderson(sale_qty, dist='norm')

print(f"Statistic: {result.statistic:.4f}")
for sl, cv in zip(result.significance_level, result.critical_values):
    print(f"Significance level {sl}%: critical value {cv}")