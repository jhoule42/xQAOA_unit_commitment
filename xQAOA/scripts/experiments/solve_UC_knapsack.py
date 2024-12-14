# Solve the UC problem with Knapsack (Brute Force solution)
#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

from ADMM.scripts.utils.utils import generate_units
from ADMM.scripts.utils.visualize import visualize_optimal_power_distribution
from ADMM.scripts.solvers.classical_solver_UC import gurobi_solver
from ADMM.scripts.solvers.classical_solver_UC import classical_power_distribution
from xQAOA.scripts.utils.kp_utils import bruteforce_knapsack

#%%
class UnitCommitmentSolver:
    def __init__(self, n_units=8, load_factor=0.5):
        """
        Initialize the Unit Commitment Solver
        
        :param n_units: Number of units to generate
        :param load_factor: Fraction of maximum power to use as load
        """
        # Generate units with their characteristics
        self.A, self.B, self.C, self.p_min, self.p_max = generate_units(N=n_units)
        
        # Calculate total load
        self.L = np.sum(self.p_max) * load_factor
        
        # Initialize storage for results
        self.results = {
            'list_cost': [],
            'list_power': [],
            'list_z_i': [],
            'list_p_i': []
        }

    def check_compatibility(self, capacity, min_weight):
        """
        Verify if the solution is valid
        
        :param p_min: Minimum power generation
        :param p_max: Maximum power generation
        :param capacity: System capacity
        :param p_i: Power generation for each unit
        :param min_weight: Minimum power generation
        :return: Boolean indicating solution validity
        """
        return capacity >= min_weight
    

    def find_min_D(self, range_D, clip_power=True):
        """
        Find the smallest D value that provides a valid solution
        
        :param range_D: Range of D values to explore
        :return: Optimal D value
        """
        valid_D_power = []
        valid_capacity = []
        for D in range_D:
            p_i = (D - self.B) / (2 * self.C)
            if clip_power:
                p_i = np.clip(p_i, self.p_min, self.p_max)

            min_weight = np.min(p_i)
            capacity = np.sum(p_i) - self.L

            # Only add valid value of D
            if self.check_compatibility(capacity, min_weight):
                valid_D_power.append(D)
            if min_weight <= capacity:
                valid_capacity.append(D)

        return valid_capacity[0]
    

    def solve_knapsack(self, range_D, clip_power=True, show_progress=True):
        """
        Solve Unit Commitment problem using Knapsack approach.

        :param range_D: Range of D values to explore.
        :param show_progress: Whether to show a progress bar for the outer loop.
        """

        # Wrap the range_D with a progress bar if show_progress is True
        iterator = tqdm(range_D, desc="Solving for D values") if show_progress else range_D

        for D in iterator:
            p_i = (D - self.B) / (2 * self.C)

            # Make sure that the values are within the optimal range
            if clip_power:
                p_i = np.clip(p_i, self.p_min, self.p_max)
            
            # Knapsack Mapping
            # ∑ v_i * y_i ==> ∑ A*y_i + B*p_i*y_i + C*(p_i*^2)*y_i
            # ∑ w_i * y_i ≤ c  ==>  ∑ p_i * z_i ≤ ∑ p_i - L
            # capacity = np.abs(np.sum(p_i) - L)
            capacity = np.sum(p_i) - self.L
            w = p_i
            v = self.A + self.B*p_i + self.C*(p_i**2)

            # Solve with Brute Force
            solutions = bruteforce_knapsack(v, w, capacity, bit_mapping='inverse', show_progress=False)
            bitstrings_ranked = [i[2] for i in solutions]
            
            # Compute costs
            z_i = [int(char) for char in bitstrings_ranked[0]]
            
            cost = np.sum(self.A*z_i + self.B*p_i*z_i + self.C*(p_i**2)*z_i)
            power = np.sum(p_i*z_i)

            # Store results
            self.results['list_cost'].append(cost)
            self.results['list_power'].append(power)
            self.results['list_p_i'].append(p_i*z_i)
            self.results['list_z_i'].append(z_i)


    def extract_quantum_results(self, range_D):
        """
        Extract the best results from quantum solver
        
        :param range_D: Range of D values used in solving
        :return: Dictionary of results
        """
        array_cost = np.array(self.results['list_cost'])

        try:        
            # Find the smallest non-zero cost
            min_array_cost = np.min(array_cost[array_cost != 0])
            min_index = np.where(array_cost == min_array_cost)[0][0]
        
            results_quantum = {
                'bitstring': ''.join(map(str, self.results['list_z_i'][min_index])),
                'power': self.results['list_p_i'][min_index],
                'cost': min_array_cost,
                'optimal_D': range_D[min_index]
            }
        except:
            results_quantum = {
                'bitstring': None,
                'power': 0,
                'cost': 0,
                'optimal_D': 0
            }
        return results_quantum

    def plot_cost_vs_d(self, range_D, title='Cost vs. D Parameter'):
        """
        Plot the cost vs D parameter
        
        :param range_D: Range of D values
        :param title: Plot title
        """
        plt.figure(figsize=(10, 8))
        plt.plot(range_D, self.results['list_cost'], 
                 marker='o', linestyle='-', markersize=5)

        plt.xlabel('D Parameter', fontsize=12)
        plt.ylabel('Cost', fontsize=12)
        plt.title(title, fontsize=14, pad=15)
        # plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()




# %%

def run_unit_commitment(load_factors, n_units=10, clip_power=True, vals_D_optimize=1):
    """
    Run Unit Commitment problem for one or multiple load factors.

    :param load_factors: Single value or list of load factors to evaluate.
    :param n_units: Number of units in the system.
    :return: Dictionary of results for each load factor.
    """

    # Ensure 'load_factors' is iterable
    if not isinstance(load_factors, (list, np.ndarray)):
        load_factors = [load_factors]

    # Dictionary to store results for each load factor
    all_results = {}

    # Adding a progress bar to the loop
    for load_factor in tqdm(load_factors, desc="Processing Load Factors", unit="factor"):

        solver = UnitCommitmentSolver(n_units=n_units, load_factor=load_factor)

        # Find minimum D
        range_D_test = np.linspace(0, 80, 5000)
        min_D = solver.find_min_D(range_D_test, clip_power)

        # Solve Knapsack
        if vals_D_optimize > 1:
            valid_range_D = np.linspace(min_D, min_D+5, vals_D_optimize)
        else:
            valid_range_D = [min_D]
        solver.solve_knapsack(valid_range_D, show_progress=False)

        # Extract quantum results
        results_quantum = solver.extract_quantum_results(valid_range_D)

        # Solve using classical method (Gurobi)
        results_gurobi = gurobi_solver(
            solver.A, solver.B, solver.C, solver.L, solver.p_min, solver.p_max
        )

        # Compute optimized quantum power distribution
        try:
            quantum_opt = classical_power_distribution(
                results_quantum["bitstring"],
                solver.A, solver.B, solver.C,
                solver.p_min, solver.p_max, solver.L
            )
            results_quantum_opt = {
                "bitstring": results_quantum["bitstring"],
                "power": quantum_opt[0],
            }
        except:
            quantum_opt = (None, np.inf)
            results_quantum_opt = {
                "bitstring": None,
                "power": 0,
            }

        # Store results for this load factor
        all_results[load_factor] = {
            "results_quantum": results_quantum,
            "results_gurobi": results_gurobi,
            "results_quantum_opt": results_quantum_opt,
            "cost_quantum_opt": quantum_opt[1],
            "optimality_ratio": (1 - (results_gurobi["cost"] / quantum_opt[1])) * 100,
        }

    return all_results


#%% ===================  RUN ALGORITHM for Multiple Loads ===================
load_factors = np.linspace(0.1, 0.9, 100)
results = run_unit_commitment(load_factors,
                              n_units=8,
                              clip_power=True,
                              vals_D_optimize=10)
print('Done.')

#%% ==================== VISUALIZE SOLUTIONS ====================

plt.figure(figsize=(10, 6))
ratio_optim = []
for load_factor, result in results.items():
    ratio_optim.append(100-result['optimality_ratio'])
    plt.scatter(load_factor, 100-result['optimality_ratio'], c='k')

plt.axhline(92, linestyle='--', c='r', linewidth=1.5)

# Axis labels and limits
plt.xlabel("Load Factor", fontsize=14)
plt.ylabel("Ratio Optimality (%)", fontsize=14)  # Adjusted label for clarity
plt.xlim(0, 1.0)
plt.ylim(bottom=0, top=102)
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(loc='upper left', fontsize=12)

# Adjusting tick sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adding a title (optional)
plt.title("Load Factor vs. Ratio Optimality", fontsize=16)

# Saving and displaying the figure
plt.tight_layout()
plt.show()

ratio_optim = np.array(ratio_optim)
ratio_optim = ratio_optim[ratio_optim != 0]
print(f"AVG RATIO OPTIM: {np.mean(ratio_optim):.2f}")


# %% Show Power distribution
def visualize_optimal_power_distribution(param_exec, show_details_cost=True, **power_distributions):
    """
    Plots the optimal power distributions from the provided dictionaries.
    Optionally shows unit parameters A, B, C for each unit under the bars.

    Args:
        param_exec (dict): Contains problem parameters like 'p_min', 'p_max', 'A', 'B', 'C'.
        show_details_cost (bool): If True, prints detailed cost calculations.
        **power_distributions: Arbitrary number of power distribution dictionaries with labels as keys.
    """
    
    # Ensure p_min and p_max are numpy arrays for consistent indexing
    p_min = np.array(param_exec['p_min'])
    p_max = np.array(param_exec['p_max'])
    A = param_exec['A']
    B = param_exec['B']
    C = param_exec['C']

    num_units = len(p_min)  # Number of power units
    unit_indices = np.arange(num_units)  # Indices for the units

    # Create the bar plot for the range from p_min to p_max
    plt.figure(figsize=(10, 6))

    # Create bars for the range from p_min to p_max
    bars = plt.bar(unit_indices, p_max - p_min, bottom=p_min, color='lightgrey',
                   edgecolor='black', linewidth=1.2, label='Power Range')

    # Initialize total cost storage for each power distribution
    total_costs = {label: 0 for label in power_distributions}

    # Plot the power distributions
    colors = ['royalblue', 'crimson', 'green', 'purple', 'orange']  # Add more colors if needed
    for idx, (label, power_array) in enumerate(power_distributions.items()):
        power_array = np.array(power_array['power'])  # Ensure it's a numpy array
        print("power array", power_array)
        units_on = (power_array > 0).astype(int)  # 1 if on, 0 if off

        # Plot the optimal power output lines
        for i in range(num_units):
            plt.hlines(y=power_array[i], xmin=bars[i].get_x(),
                       xmax=bars[i].get_x() + bars[i].get_width(),
                       color=colors[idx % len(colors)], linestyle='-', linewidth=3,
                       label=label if i == 0 else "")

        # Calculate and print detailed costs (if enabled)
        if show_details_cost:
            print(f'\n{label}')
            for i in range(num_units):
                cost = A[i] * units_on[i] + B[i] * power_array[i] + C[i] * (power_array[i] ** 2)
                total_costs[label] += cost

                # Format the output for better alignment
                print(f"Unit {i+1}: {A[i]:>6.1f} x {units_on[i]} + "
                      f"{B[i]:>6.2f} x {power_array[i]:>8.2f} + "
                      f"{C[i]:>8.3f} x {power_array[i]**2:>8.2f} = {cost:>30.2f}")

            print(f"{'Total ' + label + ' Cost:':>43} {total_costs[label]:.2f}")


    # Customize the plot
    plt.xlabel('Power Units', fontsize=14)
    plt.ylabel('Power Output', fontsize=14)
    plt.title('Optimal Power Distribution', fontsize=16, fontweight='bold')
    plt.ylim(bottom=0)
    plt.xticks(unit_indices, [f'Unit {i+1}' for i in unit_indices], fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()


# visualize_optimal_power_distribution(param_exec,
#                                      Gurobi = results_gurobi,
#                                      Quantum = results_quantum
#                                      )
