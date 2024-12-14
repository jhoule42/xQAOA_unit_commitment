""" Scrip to send jobs to the hardware to solve Knapsack. """
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

from qiskit_ibm_runtime.fake_provider import FakeAuckland
from qiskit_ibm_runtime import SamplerV2, Batch
from qiskit_ibm_runtime import QiskitRuntimeService

#%% ============================= BACKEND CONNECTION =============================
service = QiskitRuntimeService(channel="ibm_quantum", instance='pinq-quebec-hub/universit-de-cal/main')
backend = service.backend('ibm_quebec')
# backend = FakeAuckland()

if 'fake' in backend.name:
    fake_hardware = True
else:
    fake_hardware = False
print("Backend Connected.")

#%% ============================= PARAMETERS =============================
n = 127 # number of qubits
shots = 50000
k_range = [15]  # Simplified from np.arange(10, 24, 1)
theta_range = [-1]  # Simplified from [0, -0.5, -1]
N_beta, N_gamma = 15, 20  # Number of grid points for beta and gamma
beta_values = np.linspace(np.pi/2, 0, N_beta)
gamma_values = np.linspace(0, np.pi, N_gamma)
bit_mapping = 'regular' # choose 'inverse' to solve UC
PATH_RUNS = "/Users/julien-pierrehoule/Documents/Stage/T3/Code/xQAOA/runs/hardware"
pm = generate_preset_pass_manager(optimization_level=2, backend=backend)


# Format time for folder name
current_time = datetime.now()
timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

# Create execution folder
folder_name = f"KP_N{n}_GRID{N_beta*N_gamma}"
os.makedirs(f"{PATH_RUNS}/{folder_name}", exist_ok=True)
print(f"Folder created: {folder_name}")

# Save execution parameters to dict
dict_params = {}
dict_params['execution_type'] = "hardware"
dict_params['exec_time'] = timestamp
dict_params['n_units'] = n
dict_params['k_range'] = k_range
dict_params['theta_range'] = theta_range
dict_params['N_beta'] = N_beta
dict_params['N_gamma'] = N_gamma
dict_params['bit_mapping'] = bit_mapping

# List of distribution functions
list_distributions = [generate_profit_spanner]


#%% ======================= GENERATING QUANTUM CIRCUITS =======================

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

    # COPULA MIXER
    print("\nCOPULA MIXER")
    optimizer_C = QKPOptimizer(v, w, c,
                               mixer='copula',
                               generate_jobs=True,
                               run_hardware=True,
                               backend=backend,
                               pass_manager=pm)
    
    list_qc = optimizer_C.generate_circuits(k_range, theta_range, 
                                            beta_values, gamma_values)

# Save parameter to file
with open(f'{PATH_RUNS}/{folder_name}/parameters.json', 'w') as file:
    json.dump(dict_params, file, indent=4)
print('Execution parameters saved to file.')


#%% ===================== TRANSPILING QUANTUM CIRCUITS =====================
print("Transpiling Quantum Circuits.")
list_isa_qc = optimizer_C.transpile_circuits(list_qc, pass_manager=pm)

# Get list of betas and gammas parameters
list_params = [(beta, gamma) for beta in beta_values for gamma in gamma_values]

#%% ===================== SENDING JOBS TO THE HARDWARE =====================
dict_jobs_id = {}

with Batch(backend=backend, max_time='3h') as batch:
    sampler = SamplerV2(mode=batch)
    print('Sending Batch Job.')

    # Wrap the loop with tqdm for a progress bar
    for idx, isa_qc in enumerate(tqdm(list_isa_qc, desc="Submitting jobs")):
        job = sampler.run([isa_qc], shots=shots)
        job_id = job.job_id()
        dict_jobs_id[job_id] = {}

        params = list_params[idx]
        dict_jobs_id[job_id]['params'] = params

        # Extract results of fake hardware
        if fake_hardware:
            result = job.result()[0]
            counts = result.data.meas.get_counts()
            dict_jobs_id[job_id]['counts'] = counts

print('Done submitting jobs.')

with open(f'{PATH_RUNS}/{folder_name}/dict_jobs.json', 'w') as file:
    json.dump(dict_jobs_id, file, indent=4)
print('Jobs id saved to file.')
# %%