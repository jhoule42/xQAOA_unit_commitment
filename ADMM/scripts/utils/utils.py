""" Various utility function used in the UC code for computation and plotting.
Author: Julien-Pierre Houle. """

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from UC.scripts.solvers.classical_solver_UC import classical_power_distribution


def generate_units(N, A_range=(10, 50), B_range=(0.5, 1.5), C_range=(0.01, 0.2), p_min_range=(10, 20), p_max_range=(50, 100), generate_load=False):
    """
    Generate N power units with random characteristics.
    """
    A = np.random.uniform(*A_range, N)  # Linear fixed cost coefficients
    B = np.random.uniform(*B_range, N)  # Linear operational cost coefficients
    C = np.random.uniform(*C_range, N)  # Quadratic cost coefficients
    p_min = np.random.uniform(*p_min_range, N)  # Minimum power outputs
    p_max = np.random.uniform(*p_max_range, N)  # Maximum power outputs

    if generate_load:
        L = np.random.uniform(np.min(p_min), np.sum(p_max))
        return list(A), list(B), list(C), list(p_min), list(p_max), L
    else:
        return np.array(A), np.array(B), np.array(C), np.array(p_min), np.array(p_max)



def evaluate_perf_algo(counts, A, B, C, p_min, p_max, L):
    """ Evaluate the quality of the solutions provided by the QAOA algorithm.
    Provide a ranking of the optimal counts."""

    # Sorting the dictionary by values in descending order
    count_order = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    # Evaluate the optimal solution for each counts
    dict_count_perf = {}
    count_rank = {}
    for bitstring in count_order:
        power_dist, cost = classical_power_distribution(bitstring, A, B, C, p_min, p_max, L, raise_error=False)

        # Update dict with cost
        dict_count_perf[bitstring] = cost

    # Sorting the dictionary
    count_perf = dict(sorted(dict_count_perf.items(), key=lambda item: (item[1] == 0, item[1])))

    # Assign rank, but set rank to 0 if the cost is 0
    for idx, (bitstring, cost) in enumerate(count_perf.items()):
        count_rank[bitstring] = 0 if cost == 0 else idx + 1

    return count_perf, count_rank



def check_existing_files(save_path, format, filename_suffix, combine_plots=False):
    """
    Check if any target files already exist and get single user confirmation for overwrite.
    
    Parameters:
    -----------
    save_path : str
        Directory path where figures would be saved
    format : str
        File format extension
    filename_suffix : str
        Suffix to add to filenames
    combine_plots : bool
        Whether plots are combined or separate
        
    Returns:
    --------
    bool
        True if no files exist or user confirms overwrite, False otherwise
    list
        List of files that would be overwritten
    """
    existing_files = []
    
    if combine_plots:
        filename = os.path.join(save_path, f'ADMM_all_metrics{filename_suffix}.{format}')
        if os.path.exists(filename):
            existing_files.append(filename)
    else:
        plot_types = ['cost', 'lambda', 'lambda_mult', 'residuals', 'dual_residuals']
        for name in plot_types:
            filename = os.path.join(save_path, f'ADMM_{name}{filename_suffix}.{format}')
            if os.path.exists(filename):
                existing_files.append(filename)
    
    if existing_files:
        print("The following files already exist:")
        for file in existing_files:
            print(f"  - {file}")
        while True:
            response = input("Do you want to overwrite these files? (y/n): ").lower()
            if response in ['y', 'n']:
                return response == 'y', existing_files
            print("Please enter 'y' or 'n'")
    
    return True, existing_files



def save_results(filename, **kwargs):
    """Save multiple variables to a pickle file."""
    with open(filename, "wb") as file:
        pickle.dump(kwargs, file)
        print(f"Results saved to {filename}.")