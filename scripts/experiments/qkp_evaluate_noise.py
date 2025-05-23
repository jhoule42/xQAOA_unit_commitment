""" Run xQAOA algorithm to evaluate noise performance.
- Determine if we do better than only warm-start.
"""
#%%
from datetime import datetime
import numpy as np
import json
import os
import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

from xQAOA.scripts.utils.kp_utils import *
from xQAOA.scripts.solvers.qkp_solver import *

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeAuckland, FakeBrisbane
from qiskit_ibm_runtime import SamplerV2, Batch
from ADMM.scripts.solvers.classical_solver_UC import gurobi_knapsack_solver


#%% ============================= PARAMETERS =============================
n = 12 # number of items (qubits)
k_range = [15]  # Simplified from np.arange(10, 24, 1)
theta_range = [0]  # Simplified from [0, -0.5, -1]
N_beta, N_gamma = 20, 20  # Number of grid points for beta and gamma
beta_values = np.linspace(np.pi/2, 0, N_beta)
gamma_values = np.linspace(0, np.pi, N_gamma)
shots = 5000
bit_mapping = 'regular' # choose 'inverse' for UC
backend = FakeAuckland()
# backend = FakeBrisbane()
fake_hardware = True
PATH_RUNS = "/Users/julien-pierrehoule/Documents/Stage/T3/Code/xQAOA/runs/simulation"

# Transpilation options
pm = generate_preset_pass_manager(optimization_level=3,
                                  backend=backend)

# Format time for file parameters
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create folder with the timestamp
folder_name = f"KP_N{n}_GRID{N_beta*N_gamma}_NOISE_NOT_Shallow"

os.makedirs(f"{PATH_RUNS}/{folder_name}", exist_ok=True)
print(f"Folder created: {folder_name}")

# Save execution parameters to dict
dict_params = {}
dict_params['timestamp'] = timestamp
dict_params['backend'] = backend.name
dict_params['exec_time'] = timestamp
dict_params['n_units'] = n
dict_params['k_range'] = k_range
dict_params['theta_range'] = theta_range
dict_params['N_beta'] = N_beta
dict_params['N_gamma'] = N_gamma
dict_params['bit_mapping'] = bit_mapping
dict_params['shots'] = shots

# List of distribution functions
list_distributions = [generate_profit_spanner]


#%%
# Initialize a nested dictionary to store results for different methods
results = {'very_greedy': {},
           'lazy_greedy': {},
           'X': {},
           'copula': {}}

# Iterate over different distributions
for dist_func in list_distributions:
    func_name = dist_func.__name__
    print(f"\nUsing distribution: {func_name}")

    # Generate values and weights for the current distribution
    v, w = dist_func(n)
    c = np.ceil(np.random.uniform(0.25, 0.75) * sum(w)).astype(int)
    dict_params[func_name] = {}
    dict_params[func_name]['v'] = [int(i) for i in list(v)]
    dict_params[func_name]['w'] = [int(i) for i in list(w)]
    dict_params[func_name]['c'] = int(c)

    # Gurobi Solver Knapsack
    result_gurobi = gurobi_knapsack_solver(v, w, c, verbose=False,
                                        time_limit=60,
                                        optimality_gap=1e-20,
                                        feasibility_tolerance=1e-9)

    print(f"Optimization Results Gurobi -- {n} items")
    print(f"Bitstring: {result_gurobi['bitstring']}")
    print(f"Total Value: {result_gurobi['total_value']}")
    print(f"Total Weight: {result_gurobi['total_weight']}")
    print(f"Runtime: {result_gurobi['runtime']:.6f} seconds")

    value_gurobi = result_gurobi['total_value']
    value_opt = value_gurobi

    # COPULA MIXER
    print("\nCOPULA MIXER")
    optimizer_C = QKPOptimizer(v, w, c, mixer='copula',
                               generate_jobs=True,
                               run_hardware=True,
                               backend=backend,
                               pass_manager=pm)
    
    list_qc = optimizer_C.generate_circuits(k_range, theta_range,
                                            beta_values, gamma_values)

# Save parameter to file
with open(f'{PATH_RUNS}/{folder_name}/parameters.json', 'w') as file:
    json.dump(dict_params, file, indent=4)
print('Run parameters saved to file.')


#%%  ===================== Transpilating JOBS =====================
print("Transpiling Quantum Circuits.")
list_isa_qc = optimizer_C.transpile_circuits(list_qc, pass_manager=pm)
print(f'Circuit Depth: {list_isa_qc[0].depth()}')

# Get list of betas and gammas parameters
beta_values = [np.pi * i / N_beta for i in range(N_beta)]
gamma_values = [2 * np.pi * j / N_gamma for j in range(N_gamma)]
list_params = [(beta, gamma) for beta in beta_values for gamma in gamma_values]

#%% ===================== SENDING JOBS TO THE HARDWARE =====================

dict_jobs_id = {}

with Batch(backend=backend, max_time='2h') as batch:
    sampler = SamplerV2(mode=batch)

    print('Sending Batch Job.')
    for idx, isa_qc in enumerate(tqdm(list_isa_qc, desc="Submitting jobs")):

        job = sampler.run([isa_qc], shots=shots)
        job_id = job.job_id()
        dict_jobs_id[job_id] = {}

        params = list_params[idx]
        dict_jobs_id[job_id]['params'] = params

        # extract results of fake hardware
        result = job.result()[0]
        counts = result.data.meas.get_counts()
        dict_jobs_id[job_id]['counts'] = counts

print('Done submiting jobs.')

with open(f'{PATH_RUNS}/{folder_name}/dict_jobs.json', 'w') as file:
    json.dump(dict_jobs_id, file, indent=4)
print('Jobs id saved to file.')
