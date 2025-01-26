#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

from xQAOA.scripts.solvers.qkp_solver import *

#%%
# Problem instance data
v = np.array([3627, 580, 1835, 246, 364, 674, 840, 1391, 250, 193])
w = np.array([6012, 1297, 2148, 642, 678, 895, 1012, 1365, 502, 452])

# Calculate ratios v_i/w_i
ratios = v / w

# Sort indices by ratio for plotting
sorted_indices = np.argsort(ratios)
sorted_ratios = ratios[sorted_indices]

# Create data points for the plot
# We'll use the actual ratios from the data and create corresponding p_i values
# Based on the graph, p_i is 0 before r_stop and 1 after
r_stop = 0.62  # Approximately from the graph
x_points = sorted_ratios
y_points = np.where(x_points >= r_stop, 1.0, 0.0)


#$$
# Create the plot
plt.figure(figsize=(10, 6))

# Plot the step function
plt.plot(x_points, y_points, 'b-', linewidth=1.5)

# Add points at each ratio value
plt.plot(x_points, y_points, 'bo', markersize=6)

# Add r_stop annotation with larger text
plt.annotate(r'$r_{\text{stop}}$', 
             xy=(0.6, 0), 
             xytext=(r_stop-0.1, 0.1),
             arrowprops=dict(arrowstyle='->'),
             fontsize=14)

# Set the axis labels and title
plt.xlabel('$r_i = v_i/w_i$')
plt.ylabel('$p_i$')

# Set the axis limits
plt.xlim(0.35, 1.05)
plt.ylim(top=1.1)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Save the plot to the figures folder
output_path = '/Users/julien-pierrehoule/Documents/Stage/T3/Code/xQAOA/figures/LG_distribution_plot.png'
plt.savefig(output_path, dpi=300)
plt.show()


#%% plot figure 5 in the paper (Smooth Lazy Greedy)
# Define the logistic function for fitting
def logistic_fit(r, C, k, r_star):
    return 1 / (1 + C * np.exp(-k * (r - r_star)))

# Knapsack data
v = [3627, 580, 1835, 246, 364, 674, 840, 1391, 250, 193]
w = [6012, 1297, 2148, 642, 678, 895, 1012, 1365, 502, 452]
c = 10240
r = np.array(v) / np.array(w)

# Values of k to plot
k_values = [0, 5, 10, 20]

# Plotting
plt.figure(figsize=(10, 6))
for k in k_values:

    # Calculate probabilities using the logistic bias function
    solver = QKPOptimizer(v, w, c, k)
    p_i = solver.logistic_bias(k)    
    # Plot original data points
    plt.scatter(r, p_i, label=f'k = {k}')

    # Fit a curve to the data points
    popt, _ = curve_fit(lambda r, C, r_star: logistic_fit(r, C, k, r_star), r, p_i)
    C_fit, r_star_fit = popt

    # Generate smooth curve for the fit
    r_smooth = np.linspace(0, max(r), 100)
    p_smooth = logistic_fit(r_smooth, C_fit, k, r_star_fit)
    plt.plot(r_smooth, p_smooth, linestyle='--')

plt.xlabel(r'$r_i = v_i / w_i$', fontsize=12)
plt.ylabel(r'$p_i$', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot to the figures folder
output_path = '/Users/julien-pierrehoule/Documents/Stage/T3/Code/xQAOA/figures/Smooth_Lazy_Greedy_plot.png'
plt.savefig(output_path, dpi=300)
plt.show()

# %%
