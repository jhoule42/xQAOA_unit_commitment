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
from ADMM.scripts.solvers.classical_solver_UC import gurobi_knapsack_solver

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
    

    def find_min_D(self, range_D, clip_power=True):
        """
        Find the smallest D value that provides a valid solution
        
        :param range_D: Range of D values to explore
        :return: Optimal D value
        """
        min_D_valid = []

        for D in range_D:
            p_i = (D - self.B) / (2 * self.C)
            if clip_power:
                p_i = np.clip(p_i, self.p_min, self.p_max)

            min_weight = np.min(p_i)
            capacity = np.sum(p_i) - self.L

            if min_weight <= capacity:
                min_D_valid.append(D)

        # Note: there is no max value since they will be clipped if
        # they are outside the range

        return min_D_valid[0]
    
    

    def solve_knapsack(self, knapsack_solver, range_D, clip_power=True, show_progress=True):
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

            # Solve with Knapsack with either Brute Force or Gurobi
            if knapsack_solver == 'gurobi':
                solutions = gurobi_knapsack_solver(v, w, capacity, bit_mapping='inverse')
                z_i = [int(char) for char in solutions['bitstring']]
                
            else:
                solutions = bruteforce_knapsack(v, w, capacity, bit_mapping='inverse', show_progress=False)
                bitstrings_ranked = [i[2] for i in solutions]            
                z_i = [int(char) for char in bitstrings_ranked[0]]
        
            # Compute costs
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
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()




# %%

def run_unit_commitment(load_factors, kp_solver, n_units=10, clip_power=True, vals_D_optimize=1):
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
        solver.solve_knapsack(knapsack_solver=kp_solver,
                              range_D=valid_range_D,
                              show_progress=False)

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



def compute_cost_vs_D(load_factors, kp_solver='gurobi', vals_D_optimize=10, max_D=50,
                      n_units=10, clip_power=True):
    """_summary_

    Args:
        load_factors (_type_): _description_
        kp_solver (_type_): _description_
        n_units (int, optional): _description_. Defaults to 10.
        clip_power (bool, optional): _description_. Defaults to True.
    """

    # kp_solver = 'gurobi'
    # vals_D_optimize = 50
    list_cost = []

    for load in load_factors:
        solver = UnitCommitmentSolver(n_units=n_units, load_factor=load)

        # Find minimum D
        range_D_test = np.linspace(0, 100, 5000)
        min_D = solver.find_min_D(range_D_test, clip_power)

        valid_range_D = np.linspace(min_D, max_D, vals_D_optimize)

        solver.solve_knapsack(knapsack_solver=kp_solver,
                              range_D=valid_range_D,
                              clip_power=clip_power,
                              show_progress=False)
            
        # cost vs D
        list_cost.append((load, solver.results['list_cost'], valid_range_D))
            
    return list_cost


#%% ===================  RUN ALGORITHM for Multiple Loads ===================
load_factors = np.linspace(0.05, 0.98, 80)
knapsack_solver='gurobi' # 'bruforce' or 'gurobi'
n_units = 100
vals_D_optimize = 20

results = run_unit_commitment(load_factors,
                              kp_solver=knapsack_solver,
                              n_units=n_units,
                              clip_power=True,
                              vals_D_optimize=vals_D_optimize)
print('Done.')

#%% ==================== VISUALIZE SOLUTIONS ====================

plt.figure(figsize=(12, 4))
ratio_optim = []
for load_factor, result in results.items():
    ratio_optim.append(100-result['optimality_ratio'])
    plt.scatter(load_factor, 100-result['optimality_ratio'], c='k', edgecolor='w',\
                s=100, alpha=0.8, label=r'Nb of $\mathcal{D}$ values optimized: '+str(vals_D_optimize)
                if load_factor == list(results.keys())[0] else "")

# Axis labels and limits
plt.xlabel("Load Factor", fontsize=16)
plt.ylabel("Ratio Optimality (%)", fontsize=16)
plt.xlim(0., 1.0)
plt.ylim(95, 100.4)
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=14, loc='lower right')

# Adjusting tick sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Saving and displaying the figure
plt.tight_layout()
plt.savefig(f"../../figures/UC_optim_ratio_{n_units}u_{vals_D_optimize}-vals_D.png", dpi=300,
            bbox_inches='tight')
plt.show()

ratio_optim = np.array(ratio_optim)
ratio_optim = ratio_optim[ratio_optim != 0]
print(f"AVG RATIO OPTIM: {np.mean(ratio_optim):.2f}")

# %%
n_units = 100
list_cost = compute_cost_vs_D(load_factors=np.arange(0.05, 0.96, 0.15),
                              kp_solver='gurobi',
                              vals_D_optimize=500,
                              max_D=40,
                              n_units=n_units,
                              clip_power=True)

#%%
def plot_cost_vs_D_for_loads(list_cost):
    """
    Plot the cost vs D parameter for different load factors with enhanced styling.

    :param list_cost: List of tuples containing load factor and corresponding costs.
    """
    plt.figure(figsize=(12, 8))
    
    # Color palette  
    colors = plt.cm.viridis(np.linspace(0, 1, len(list_cost)))
    
    for (load, costs, valid_range_D), color in zip(list_cost, colors):
        plt.plot(valid_range_D, costs, label=f'Load Factor: {load:.2f}', 
                linewidth=2.5, color=color, alpha=0.8)
    
    plt.xlabel(r'$\mathcal{D}$', fontsize=16, labelpad=10)
    plt.ylabel('Cost', fontsize=16, labelpad=10)
    
    plt.xlim(0, 30)
    plt.yscale("log")
    
    # Customize grid
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # Customize ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Legend inside the plot
    plt.legend(fontsize=12, loc='upper left')
    
    # Saving and displaying the figure
    plt.tight_layout()
    plt.savefig(f"../../figures/UC_cost_vs_D_{n_units}units.png", dpi=300,
                bbox_inches='tight')
    plt.show()

# Plot the cost vs D for different load factors
plot_cost_vs_D_for_loads(list_cost)
# %%
