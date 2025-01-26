""" Run xQAOA algorithm on a fake hardware backend."""
#%%
from datetime import datetime
import numpy as np
import json
import os
import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2, FakeQuebec
from qiskit_ibm_runtime import SamplerV2, Batch

from xQAOA.scripts.utils.kp_utils import *
from xQAOA.scripts.solvers.qkp_solver import *


#%% ============================= PARAMETERS =============================
n = 6  # number of items (qubits)
shots = 2000
backend = FakeGuadalupeV2()
# backend = FakeQuebec()

k_range = [15]  # Simplified from np.arange(10, 24, 1)
theta_range = [-1]  # Simplified from [0, -0.5, -1]
N_beta, N_gamma = 20, 20  # Number of grid points for beta and gamma
bit_mapping = 'regular' # choose 'inverse' for UC
func_distribution = generate_strongly_correlated  # Knapsack distribution function
load_factor = None  # if set to None; random load between [0.25-0.75] max weight
fake_hardware = True
warm_start_only = False
mixer = 'copula_2'

# Specified an experiment name, to run the same Knapsack problem.
# Else, set to None
# kp_from_experiments = f"KP_N12_GRID64_copula_shallow"
kp_from_experiments = None

# Transpilation options
pm = generate_preset_pass_manager(optimization_level=3,
                                  backend=backend)
PATH_RUNS = "/Users/julien-pierrehoule/Documents/Stage/T3/Code/xQAOA/runs/simulation"


# Format time for file parameters
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create execution folder
folder_name = f"KP_N{n}_GRID{N_beta*N_gamma}_{mixer}_{func_distribution.__name__}"
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


#%% ======================= GENERATING QUANTUM CIRCUITS =======================

if kp_from_experiments:
    with open(f'{PATH_RUNS}/{kp_from_experiments}/parameters.json', 'r') as file:
        dict_params = json.load(file)

    # !! Need to generalize profit distributin name !!
    # v = dict_params['generate_profit_spanner']['v']
    # w = dict_params['generate_profit_spanner']['w']
    # c = dict_params['generate_profit_spanner']['c']
    v = dict_params['v']
    w = dict_params['w']
    c = dict_params['c']
    print(f'Retreived Knapsack Parameters: {kp_from_experiments}')

else:
    # Generate values and weights for the current distribution
    v, w = func_distribution(n)
    if load_factor:
        c = np.ceil(np.random.uniform(load_factor) * sum(w)).astype(int)
    else:
        c = np.ceil(np.random.uniform(0.25, 0.75) * sum(w)).astype(int)

    dict_params['v'] = [int(i) for i in list(v)]
    dict_params['w'] = [int(i) for i in list(w)]
    dict_params['c'] = int(c)
    dict_params['func_distribution'] = func_distribution.__name__
    print(f'Created Knapsack Parameters from {func_distribution.__name__}.')


print("Generate Quantum Circuits.")
optimizer_C = QKPOptimizer(v, w, c,
                            mixer=mixer,
                            generate_jobs=True,
                            run_hardware=True,
                            backend=backend,
                            pass_manager=pm)

list_qc = optimizer_C.generate_circuits(k_range,
                                        theta_range, 
                                        beta_values=np.linspace(np.pi/2, 0, N_beta),
                                        gamma_values=np.linspace(0, np.pi, N_gamma),
                                        warm_start_only=warm_start_only)
print(f"Logical circuit depth: {list_qc[0].depth()}")

# Save parameter to file
with open(f'{PATH_RUNS}/{folder_name}/parameters.json', 'w') as file:
    json.dump(dict_params, file, indent=4)
print('Execution parameters saved to file.')


#%%  ===================== Transpilating JOBS =====================
print("Transpiling Quantum Circuits.")
list_isa_qc = optimizer_C.transpile_circuits(list_qc, pass_manager=pm)
print(f"Transpile circuit depth: {list_isa_qc[0].depth()}")

# Get list of betas and gammas parameters
list_params = [(beta, gamma) for beta in np.linspace(np.pi/2, 0, N_beta) \
                             for gamma in np.linspace(0, np.pi, N_gamma)]

#%% ===================== SENDING JOBS TO THE HARDWARE =====================

dict_jobs_id = {}

with Batch(backend=backend, max_time='2h') as batch:
    sampler = SamplerV2(mode=batch)
    workload_id = batch.session_id

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
# %%