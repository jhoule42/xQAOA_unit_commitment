""" xQAOA quantum simulator to solve Knapsack. """
#%%
import numpy as np
from datetime import datetime
import os
import json
import sys

sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")
from xQAOA.scripts.solvers.qkp_solver import *
from xQAOA.scripts.utils.kp_utils import *
from xQAOA.scripts.utils.visualize import *
from ADMM.scripts.utils.visualize import *  # check if I still need it


#%% ==================== PARAMETERS ====================
n = 8  # nb of qubits (items)
k_range = [15] 
theta_range = [-1]  # Copula mixer parameter
N_beta, N_gamma = 20, 20  # Number of grid points for beta and gamma
shots = 3000
bit_mapping = 'regular'
PATH_RUNS = "/Users/julien-pierrehoule/Documents/Stage/T3/Code/xQAOA/runs/simulation"

# Initialize a nested dictionary to store results for different methods
results = {
    'very_greedy': {},
    'lazy_greedy': {},
    'copula': {},
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
range_capacity_ratio = np.linspace(0.2, 0.8, 10)

# Create a list to store results for each capacity ratio
all_results = []

for c_ratio in range_capacity_ratio:
    print(f"\nCapacity Ratio: {c_ratio}")

    # Dictionary to store results for this specific c_ratio
    ratio_results = {
        'c_ratio': c_ratio,
        'distribution_results': []
    }

    # Iterate over different distributions
    for dist_func in list_distributions:
        print(f"\nUsing distribution: {dist_func.__name__}")

        # Generate values and weights for the current distribution
        v, w = dist_func(n)
        c = np.ceil(c_ratio * sum(w)).astype(int)

        # Solve with Brute Force for optimal solution
        solutions = bruteforce_knapsack(v, w, c)
        bitstrings_ranked = [i[2] for i in solutions]
        optimal_value = solutions[0][0]
        print(f"\nOptimal Solution (BruteForce): {optimal_value}")


        # Run Lazy Greedy Knapsack
        value_LG, weight_LG, bitstring_LG = lazy_greedy_knapsack(v, w, c)
        results['lazy_greedy'][dist_func.__name__] = {
                'ratio_optim': value_LG / optimal_value,
                'rank_solution': bitstrings_ranked.index(bitstring_LG) + 1}
        print(f"Lazy Greedy value: {value_LG}")


        # Run Very Greedy Knapsack
        value_VG, weight_VG, bitstring_VG = very_greedy_knapsack(v, w, c)
        results['very_greedy'][dist_func.__name__] = {
                'ratio_optim': value_VG / optimal_value,
                'rank_solution': bitstrings_ranked.index(bitstring_VG) + 1}
        print(f"Very Greedy value: {value_VG}")


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
        
        optimizer_C.parameter_optimization(k_range, theta_range,
                                           N_beta, N_gamma,
                                           bit_mapping='regular',
                                           shots=shots)

        # Store results in dict
        results['copula'][dist_func.__name__] = {}
        results['copula'][dist_func.__name__] = {
                'ratio_optim': optimizer_C.best_value / optimal_value,
                'rank_solution': bitstrings_ranked.index(optimizer_C.best_bitstring) + 1,
                'beta_opt': optimizer_C.best_params[0],
                'gamma_opt': optimizer_C.best_params[1],
                'best_value': optimizer_C.best_value}
        
        if (optimizer_C.best_value / optimal_value) > 0.95:
            list_opt_parameters.append(optimizer_C.best_params)

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

# Show optimal parameters distribution
plot_optimal_parameters(list_opt_parameters)


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
plot_value_distribution(
    combined_data,
    optimal_value=solutions[0][0],
    best_val_found=results['copula']['generate_profit_spanner']['best_value'])
# %%