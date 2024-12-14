# Run QKP and produce a heatmaps to determine the optimal parameters (beta, gamma)
# The idea was to see if it's possible to reproduce the figure from Natalie paper.
#%%
import numpy as np
from datetime import datetime
import os
import json
import sys

sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")
from xQAOA.scripts.utils.kp_utils import *
from xQAOA.scripts.solvers.qkp_solver import *
from UC.scripts.utils.visualize import plot_custom_histogram, plot_value_distribution


#%% ==================== PARAMETERS ====================
n = 18
k_range = [15]  # Simplified from np.arange(10, 24, 1)
theta_range = [-1]  # Simplified from [0, -0.5, -1]
N_beta, N_gamma = 30, 30  # Number of grid points for beta and gamma
shots = 10000
bit_mapping = 'regular'
PATH_RUNS = "/Users/julien-pierrehoule/Documents/Stage/T3/Code/xQAOA/runs/simulation"

# Initialize a nested dictionary to store results for different methods
results = {
    'very_greedy': {},
    'lazy_greedy': {},
    'copula': {},
    # 'X': {}
}


# Format time for folder name
current_time = datetime.now()
timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

# Create folder
folder_name = f"KP_N{n}_GRID{N_beta*N_gamma}_SIM"
os.makedirs(f"{PATH_RUNS}/{folder_name}", exist_ok=True)
print(f"Folder created: {folder_name}")

# Save parameters to dict
dict_params = {}
dict_params['execution_type'] = "simulator"
dict_params['exec_time'] = timestamp
dict_params['n_units'] = n
dict_params['k_range'] = k_range
dict_params['theta_range'] = theta_range
dict_params['N_beta'] = N_beta
dict_params['N_gamma'] = N_gamma
dict_params['bit_mapping'] = bit_mapping

# List of distribution functions
list_distributions = [generate_profit_spanner]

# Save parameter to file
with open(f'{PATH_RUNS}/{folder_name}/parameters.json', 'w') as file:
    json.dump(dict_params, file, indent=4)
print('Run parameters saved to file.')


#%% ==================== RUN SIMULATION ====================

list_opt_parameters = []
range_capacity_ratio = np.linspace(0.2, 0.8, 20)

# Create a list to store results for each capacity ratio
all_results = []

for c_ratio in [0.3]:
    print(f"Capacity Ratio: {c_ratio}")

    # Dictionary to store results for this specific c_ratio
    ratio_results = {
        'c_ratio': c_ratio,
        'distribution_results': []
    }
    

    # Iterate over different distributions
    for dist_func in list_distributions:
        # print(f"\nUsing distribution: {dist_func.__name__}")

        # Generate values and weights for the current distribution
        v, w = dist_func(n)
        c = np.ceil(c_ratio * sum(w)).astype(int)

        # Solve with Brute Force for optimal solution
        solutions = bruteforce_knapsack(v, w, c)
        bitstrings_ranked = [i[2] for i in solutions]
        optimal_value = solutions[0][0]
        # print(f"\nOptimal Solution (BruteForce): {optimal_value}")


        # Run Lazy Greedy Knapsack
        value_LG, weight_LG, bitstring_LG = lazy_greedy_knapsack(v, w, c)
        results['lazy_greedy'][dist_func.__name__] = {
                'ratio_optim': value_LG / optimal_value,
                'rank_solution': bitstrings_ranked.index(bitstring_LG) + 1}
        # print(f"Lazy Greedy value: {value_LG}")


        # Run Very Greedy Knapsack
        value_VG, weight_VG, bitstring_VG = very_greedy_knapsack(v, w, c)
        results['very_greedy'][dist_func.__name__] = {
                'ratio_optim': value_VG / optimal_value,
                'rank_solution': bitstrings_ranked.index(bitstring_VG) + 1}
        # print(f"Very Greedy value: {value_VG}")


        # # X (Standard) MIXER
        # print("\nX MIXER")
        # optimizer_X = QKPOptimizer(v, w, c, mixer='X', optimal_solution=optimal_value)
        # optimizer_X.parameter_optimization(k_range, [0], N_beta, N_gamma)

        # # Store results in dict
        # results['X'][dist_func.__name__] = {
        #         'ratio_optim': optimizer_X.best_value / optimal_value,
        #         'rank_solution': bitstrings_ranked.index(optimizer_X.best_bitstring) + 1,
        #         'beta_opt': optimizer_X.best_params[0],
        #         'gamma_opt': optimizer_X.best_params[1],
        #         'best_value': optimizer_X.best_value}


        # COPULA MIXER
        print("\nCOPULA MIXER")
        optimizer_C = QKPOptimizer(v, w, c, mixer='copula', optimal_solution=optimal_value,
                                   speedup_computation=False)
        optimizer_C.parameter_optimization(k_range, theta_range, N_beta, N_gamma,
                                           bit_mapping='regular', shots=shots)


        # # Store results in dict
        # results['copula'][dist_func.__name__] = {}
        # results['copula'][dist_func.__name__] = {
        #         'ratio_optim': optimizer_C.best_value / optimal_value,
        #         'rank_solution': bitstrings_ranked.index(optimizer_C.best_bitstring) + 1,
        #         'beta_opt': optimizer_C.best_params[0],
        #         'gamma_opt': optimizer_C.best_params[1],
        #         'best_value': optimizer_C.best_value}
        
        # if (optimizer_C.best_value / optimal_value) > 0.95:
        #     list_opt_parameters.append(optimizer_C.best_params)

        # Store detailed results for this distribution
        dist_result = {
            'distribution': dist_func.__name__,
            'beta': optimizer_C.best_params[0],
            'gamma': optimizer_C.best_params[1],
            'performance': optimizer_C.best_value / optimal_value
        }
        
        ratio_results['distribution_results'].append(dist_result)
    
    all_results.append(ratio_results)

print('Done.')

#%% ==================== VISUALIZE RESULTS ====================

nb_optimal = 0
dict_params = optimizer_C.dict_all_parameters
for params in dict_params:
    ratio = dict_params[params]/optimal_value

    if ratio >= 0.98 and ratio <= 1.0:
        nb_optimal += 1
print("Nb optimal parameters", nb_optimal)

#%%


# Show optimal parameters distribution
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import gaussian_filter

def create_2d_knapsack_heatmap(results_dict, optimal_value, sigma=1):
    # Extract unique beta and gamma values
    beta_values = sorted(set(float(key.split(',')[0]) for key in results_dict.keys()))
    gamma_values = sorted(set(float(key.split(',')[1]) for key in results_dict.keys()))
    
    # Create a 2D grid to store values
    value_grid = np.zeros((len(gamma_values), len(beta_values)))
    
    # Fill the grid with Knapsack values
    for key, value in results_dict.items():
        beta, gamma = map(float, key.split(','))
        beta_index = beta_values.index(beta)
        gamma_index = gamma_values.index(gamma)
        
        # Modify value based on ratio condition
        ratio = value / optimal_value
        value_grid[gamma_index, beta_index] = 0 if ratio > 1.0 or ratio < 0. else ratio
    
    # Apply Gaussian smoothing
    smoothed_grid = gaussian_filter(value_grid, sigma=sigma)
    
    # Create the plot
    plt.figure(figsize=(18, 6))
    
    # Use seaborn heatmap for better visualization
    ax = sns.heatmap(smoothed_grid,
        cmap='plasma',
        cbar_kws={'label': 'Smoothed Performance Ratio'},
        # Select a subset of ticks for better readability
        xticklabels=[f'{b:.2f}' for i, b in enumerate(beta_values) if i % 3 == 0],
        yticklabels=[f'{g:.2f}' for i, g in enumerate(gamma_values) if i % 3 == 0]
    )
    
    # Adjust tick positions to match selected labels
    x_tick_positions = [i for i, b in enumerate(beta_values) if i % 3 == 0]
    y_tick_positions = [i for i, g in enumerate(gamma_values) if i % 3 == 0]
    
    ax.set_xticks(x_tick_positions)
    ax.set_yticks(y_tick_positions)
    
    plt.title('Smoothed Performance Ratio for Beta and Gamma Combinations')
    plt.xlabel('Beta')
    plt.ylabel('Gamma')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()

# Usage with optional sigma parameter
# sigma controls the amount of smoothing
# Higher sigma = more smoothing
create_2d_knapsack_heatmap(optimizer_C.dict_all_parameters, optimal_value, sigma=0)

# Usage
# create_2d_knapsack_heatmap(optimizer_C.dict_all_parameters, optimal_value)


#%%
# Alternative plot using matplotlib for more customization
def create_2d_knapsack_scatter(results_dict, optimal_value):
    # Extract beta, gamma, and values
    betas = []
    gammas = []
    values = []
    
    for key, value in results_dict.items():
        beta, gamma = map(float, key.split(','))
        betas.append(beta)
        gammas.append(gamma)
        ratio = value / optimal_value
        if ratio > 1.0 or ratio < 0.0:
            values.append(0)
        else:
            values.append(ratio)
        
    
    plt.figure(figsize=(14, 6))
    
    # Create scatter plot
    scatter = plt.scatter(betas, gammas, c=values, cmap='inferno', 
                          s=100,  # Size of points
                          alpha=0.99)
    
    plt.colorbar(scatter, label='Knapsack Value')
    
    plt.title('Knapsack Values for Different Beta and Gamma Combinations')
    plt.ylabel('Gamma')
    plt.xlabel('Beta')
    
    plt.tight_layout()
    plt.show()

create_2d_knapsack_scatter(optimizer_C.dict_all_parameters, optimal_value)


#%%
# # Plot the results
# plot_rank_and_ratio(results, methods=['lazy_greedy', 'very_greedy', 'copula'],
#                     labels = ['LG', 'VG', r'$QKP_{COP}$'])

#%%
# Example of running a single instance with specific parameters
bitstring, value, weights, counts, success = optimizer_C.QKP(
    gamma=results['copula']['generate_profit_spanner']['gamma_opt'], 
    beta=results['copula']['generate_profit_spanner']['beta_opt'],
    k=15,
    theta=0)

# Prepare the combined data
combined_data = []
for value, weight, bitstring in solutions:
    # Get the count from counts dictionary; use 0 if the bitstring is not found
    bitstring_count = counts.get(bitstring, 0)
    # Create the tuple (value, count, bitstring) and add it to the list
    combined_data.append((value, bitstring_count, bitstring))

print(f"Bitstring: {bitstring}")
print(f"Value: {value}")
print(f"Weights: {weights}")
plot_custom_histogram(counts, max_bitstrings=32000, remove_xticks=True, display_text=False)


#%%
# This is usefull when trying to scale and we are limited by shots number
plot_value_distribution(combined_data,
                        optimal_value=solutions[0][0],
                        best_val_found=results['copula']['generate_profit_spanner']['best_value'])
# %%