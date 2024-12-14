#%% Run the algorithm to solve the UC problem using Knapsack. Optimize the parameter D.
# Need to transfer this to solve_UC_knapsack.py file !!!

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

from UC.scripts.utils.utils import *
from UC.scripts.utils.visualize import *
from UC.scripts.solvers.classical_solver_UC import *

from xQAOA.scripts.utils.kp_utils import *
from xQAOA.scripts.solvers.qkp_solver import *

#%% ============= Solve for different units size with same load =============
range_n_units = np.arange(6, 18, 3)
range_n_units = [8]
neasted_lit_cost = []
list_min_D = []
power_load_ratio = 0.5 # ie. 50% of power loads


for n_units in range_n_units:

    A, B, C, p_min, p_max = generate_units(N=n_units)
    L = np.sum(p_max) * power_load_ratio
    param_exec = {"L": L, "n_units":n_units,
                "A": A, "B": B, "C": C,
                "p_min": p_min, "p_max": p_max}

    list_cost = []
    list_power = []
    list_z_i = []
    list_p_i = []

    results = {
        'very_greedy': {},
        'lazy_greedy': {},
        'hourglass': {},
        'copula': {},
        'X': {}
    }

    def check_compatibilty(p_min, p_max, capacity, p_i, min_weight):
        """Make sure the solution is valid."""
        return capacity >= min_weight
    

    def find_min_D(range_D, A, B, C, L):
        """
        Finds the smallest value of D from range_D such that capacity >= 0.
        
        Parameters:
        range_D: list or np.array of possible D values
        B: constant B in the equation
        C: constant C in the equation
        L: threshold value for capacity
        
        Returns:
        optimal_D: The smallest D that satisfies the condition or None if not found
        """
        for D in range_D:
            # Calculate p_i for current D
            p_i = (D - B) / (2 * C)
            min_weight= np.min(p_i)
            capacity = np.sum(p_i) - L

            # Check if solution is compatible
            if check_compatibilty(p_min, p_max, capacity, p_i, min_weight):
                return D
            else:
                print("No suitable D found.")
                continue
            
        return None

    min_D = find_min_D( np.linspace(2, 3, 100), A, B, C, L)
    range_D = np.linspace(1, 6, 500)
    # range_D = [2]


    # Compute for different values of D
    for idx, D in enumerate(range_D):
        p_i = (D - B) / (2*C)
        list_p_i.append(p_i)
        
        # Knapsack Mapping
        # ∑ v_i * y_i ==> ∑ A*y_i + B*p_i*y_i + C*(p_i*^2)*y_i
        # ∑ w_i * y_i ≤ c  ==>  ∑ p_i * z_i ≤ ∑ p_i - L
        # capacity = np.abs(np.sum(p_i) - L)
        capacity = np.sum(p_i) - L
        w = p_i
        v = A + B*p_i + C*(p_i**2)
        print('Weights', w)
        print('Capacity', capacity)

        # Solve with Brute Force for optimal solution
        solutions = bruteforce_knapsack(v, w, capacity, bit_mapping='inverse')
        bitstrings_ranked = [i[2] for i in solutions]
        optimal_value = solutions[0][0]
        # print(f"\nOptimal Solution (BruteForce): {optimal_value}")
        # print(f"Optimal bitstring: {bitstrings_ranked[0]}")

        # Store results in dict
        results['bruteforce'] = {
                'bitstring': bitstrings_ranked[0],
                'ratio_optim': optimal_value / optimal_value,
                'rank_solution': bitstrings_ranked.index(bitstrings_ranked[0]) + 1}

        # Comput Costs
        z_i = bitstrings_ranked[0]
        z_i = [int(char) for char in z_i]

        cost = np.sum(A*z_i + B*p_i*z_i + C*(p_i**2)*z_i)
        power = np.sum(p_i*z_i) # total power delivered

        list_cost.append(cost)
        list_power.append(power)
        list_p_i.append(p_i*z_i)
        list_z_i.append(z_i)

    neasted_lit_cost.append(list_cost)
    list_min_D.append(min_D)
    print("Done.")


#%% ================== Visualize COST VS D ==================

plt.figure(figsize=(10, 8))
colors = ['k', 'blue', 'green', 'pink']
for i in range(len(neasted_lit_cost)):
    plt.plot(range_D, neasted_lit_cost[i], 
             label=f'Units: {range_n_units[i]}',
             marker='o', linestyle='-', markersize=5,
             color=colors[i])
    
    plt.axvline(list_min_D[i], label='min D', color=colors[i])

plt.xlabel('D Parameter', fontsize=12)
plt.ylabel('Cost', fontsize=12)
plt.title('Cost vs. D Parameter for Different Units', fontsize=14, pad=15)

# Add a legend
plt.legend(title='Number of Units', fontsize=10)
plt.yscale('log')
# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#%% ===================  Solving UC for different load levels  ===================

n_units = 8
neasted_lit_cost = []
range_load_level = [0.05, 0.3]

for load_level in range_load_level:

    A, B, C, p_min, p_max = generate_units(N=n_units)
    L = np.sum(p_max) * load_level # make power load midway

    list_cost = []
    list_power = []
    list_z_i = []
    list_p_i = []

    results = {
        'very_greedy': {},
        'lazy_greedy': {},
        'hourglass': {},
        'copula': {},
        'X': {}
    }

    range_D = np.linspace(1, 10, 100)

    for idx, D in enumerate(range_D):
        p_i = (D - B) / (2*C)
        
        # Knapsack Mapping
        # ∑ v_i * y_i ==> ∑ A*y_i + B*p_i*y_i + C*(p_i*^2)*y_i
        # ∑ w_i * y_i ≤ c  ==>  ∑ p_i * z_i ≤ ∑ p_i - L
        # capacity = np.abs(np.sum(p_i) - L)
        capacity = np.sum(p_i) - L
        w = p_i
        v = A + B*p_i + C*(p_i**2)

        # Solve with Brute Force for optimal solution
        solutions = bruteforce_knapsack(v, w, capacity, bit_mapping='inverse')
        bitstrings_ranked = [i[2] for i in solutions]
        optimal_value = solutions[0][0]

        # Store results in dict
        results['bruteforce'] = {
                'bitstring': bitstrings_ranked[0],
                # 'ratio_optim': optimal_value / optimal_value,
                'rank_solution': bitstrings_ranked.index(bitstrings_ranked[0]) + 1}

        # Comput Costs
        z_i = bitstrings_ranked[0]
        z_i = [int(char) for char in z_i]

        cost = np.sum(A*z_i + B*p_i*z_i + C*(p_i**2)*z_i)
        power = np.sum(p_i*z_i) # total power delivered

        list_cost.append(cost)
        list_power.append(power)
        list_p_i.append(p_i*z_i)
        list_z_i.append(z_i)

    neasted_lit_cost.append(list_cost)
    print("Done.")


#%% ================== Visualize COST VS D ==================

plt.figure(figsize=(10, 8))
for i in range(len(neasted_lit_cost)):
    plt.plot(range_D, neasted_lit_cost[0],
             label=f'Load Level: {range_load_level[i]}',
             marker='o', linestyle='-', markersize=5)

plt.xlabel('D Parameter', fontsize=12)
plt.ylabel('Cost', fontsize=12)
plt.title(f'Cost vs. D Parameter for Different Load Level | {n_units} units', fontsize=14, pad=15)

# Add a legend
plt.legend(title='Number of Units', fontsize=10)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#%%

def extract_quantum_results(list_cost, list_z_i, list_p_i, range_D):
    array_cost = np.array(list_cost)
    
    # Find the smallest non-zero cost
    min_array_cost = np.min(array_cost[array_cost != 0])
    
    # Get the index of the smallest non-zero cost
    min_index = np.where(array_cost == min_array_cost)[0][0]
    print(f"P_i", list_p_i[min_index])
    
    # Create a dictionary to store the results, similar to results_gurobi
    results_quantum = {
        'bitstring': ''.join(map(str, list_z_i[min_index])),
        'power': list_p_i[min_index],
        'cost': min_array_cost,
        'optimal_D': range_D[min_index]
    }
    
    return results_quantum

# Call the function and print the results
results_quantum = extract_quantum_results(list_cost, list_z_i, list_p_i, range_D)
print('Quantum Solver')
print(f"Cost: {results_quantum['cost']:.1f}")
print(f"Power Load: {np.sum(results_quantum['power']):.1f}/{L}")
print(f"Bitstring: {results_quantum['bitstring']}")
print('Power:', results_quantum['power'])
print(f"Optimal D: {results_quantum['optimal_D']}")

# Compare with classical solution
results_gurobi = gurobi_solver(A, B, C, L, p_min, p_max)
print("\n\nClassical Solver (Gurobi)")
print(f"Cost: {results_gurobi['cost']:.1f}")
print(f"Power Load: {np.sum(results_gurobi['power']):.1f}/{L}")
print("Bitstring:", results_gurobi['bitstring'])


#%% Visualize Power Distribution
visualize_optimal_power_distribution(param_exec,
                                     Gurobi = results_gurobi,
                                     Quantum = results_quantum
                                     )


# %%s
