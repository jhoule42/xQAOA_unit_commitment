#%%
import pickle
from UC.scripts.utils.utils import *
import matplotlib.pyplot as plt

#%%
def load_results(filename):
    """Load results from a pickle file."""
    with open(filename, "rb") as file:
        data = pickle.load(file)
        print(f"Results loaded.")
        print(f"Description: {data['run_description']}\n")
    return data

# Load the saved data
data = load_results("results/admm_results.pkl")

# Access the stored variables
run_description = data['run_description']
param_exec = data['param_exec']
result_gurobi = data['result_gurobi']
result_CS = data['result_CS']
result_BQ = data['result_BQ']
result_AQ = data['result_AQ']
value_history = data['value_history'] 
param_history = data['param_history']
gammas_history = data['gammas_history'] 
betas_history = data['betas_history'] 
power_dist_BQ = data['power_dist_BQ']


power_distrib_gurobi = result_gurobi[1]

n_units = len(param_exec['A'])
x_sol = result_AQ.x[:n_units] # extract binary string
power_dist_AQ = result_AQ.x[n_units:] # extract power distribution
power_dist_AQ = np.where(np.abs(power_dist_AQ) < 1e-3, 0, power_dist_AQ)


x_sol = result_AQ.x[:n_units] # extract binary string
power_dist_BQ = result_BQ.x[n_units:] # extract power distribution
power_dist_BQ = np.where(np.abs(power_dist_BQ) < 1e-3, 0, power_dist_BQ)

x_sol = result_CS.x[:n_units] # extract binary string
power_dist_CS = result_BQ.x[n_units:] # extract power distribution
power_dist_CS = np.where(np.abs(power_dist_CS) < 1e-3, 0, power_dist_CS)


# %%

plot_optimal_power_distribution(param_exec,
                                Gurobi = power_distrib_gurobi,
                                Advance_Quantum = power_dist_AQ,
                                ADMM_Classical = power_dist_CS,
                                Basic_Quantum = power_dist_BQ,
                                )
# %%
