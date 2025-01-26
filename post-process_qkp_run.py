""" Script to post-process the results from hardware execution. """
#%%
import json
import sys
import numpy as np
from tqdm import tqdm
from qiskit_ibm_runtime import QiskitRuntimeService

sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")
from xQAOA.scripts.utils.kp_utils import *
from xQAOA.scripts.utils.visualize import *
from ADMM.scripts.solvers.classical_solver_UC import gurobi_knapsack_solver


#%% ===================== Connect to the Backend =====================

service = QiskitRuntimeService(channel="ibm_quantum",
                               instance='pinq-quebec-hub/universit-de-cal/main')
print("Connected to the backend.")


#%% ========================  PARAMETERS  =========================

PATH_RUNS = "/Users/julien-pierrehoule/Documents/Stage/T3/Code/xQAOA/runs"
func_name = "generate_strongly_correlated"
exec_name = f"simulation/KP_N6_GRID400_copula_2_generate_strongly_correlated"  # execution directory

# need to write this to execution file !!!
workload_id = "cxq1mhjwk6yg008hmktg" # IBM Service jobs id (only for hardware run)

# Load Parameters
with open(f'{PATH_RUNS}/{exec_name}/parameters.json', 'r') as file:
    dict_params = json.load(file)

# v = np.array(dict_params['generate_profit_spanner']['v'])
# w = np.array(dict_params['generate_profit_spanner']['w'])
# c = dict_params['generate_profit_spanner']['c']
v = np.array(dict_params['v'])
w = np.array(dict_params['w'])
c = dict_params['c']

backend_name = dict_params['backend']
# backend_name = 'ibm_quebec'/

n_units = len(v)
fake_hardware = 'fake' in backend_name
N_beta = dict_params['N_beta']
N_gamma = dict_params['N_gamma']
bit_mapping = dict_params['bit_mapping']
k_range = dict_params['k_range']
theta_range = dict_params['theta_range']

# Initialize a nested dictionary to store results for different methods
results = {'very_greedy': {},
           'lazy_greedy': {},
           'copula': {},}

# Load Jobs ID
with open(f'{PATH_RUNS}/{exec_name}/dict_jobs.json', 'r') as file:
    dict_jobs = json.load(file)
print("Jobs Loaded.")


#%% ===================== CLASSICAL SOLUTIONS =====================

# Gurobi Solver Knapsack
result_gurobi = gurobi_knapsack_solver(v, w, c, verbose=False,
                                       time_limit=60,
                                       optimality_gap=1e-20,
                                       feasibility_tolerance=1e-9)

print(f"Optimization Results Gurobi -- {n_units} items")
print(f"Bitstring: {result_gurobi['bitstring']}")
print(f"Total Value: {result_gurobi['total_value']}")
print(f"Total Weight: {result_gurobi['total_weight']}")
print(f"Runtime: {result_gurobi['runtime']:.6f} seconds")

value_gurobi = result_gurobi['total_value']
value_opt = value_gurobi


# Solve with Brute Force for optimal solution for small KP
if n_units <= 25:
    bruteforce_bitstring = bruteforce_knapsack(v, w, c)
    bitstrings_ranked = [i[2] for i in bruteforce_bitstring]
    bruteforce_value = bruteforce_bitstring[0][0]
    print(f"\nOptimal Solution (BruteForce): {bruteforce_value}")

    # Update the Best Value if BF was computed
    if bruteforce_value > value_gurobi:
        value_opt = bruteforce_value

# Compute Lazy Greedy Knapsack
value_LG, weight_LG, bitstring_LG = lazy_greedy_knapsack(v, w, c)
print(f"Lazy Greedy value: {value_LG}")
results['lazy_greedy'][func_name] = {
        'value': value_LG,
        'weight': weight_LG,
        'bitstring': bitstring_LG,
        'ratio_optim': value_LG / value_opt
        }

# Compute Very Greedy Knapsack
value_VG, weight_VG, bitstring_VG = very_greedy_knapsack(v, w, c)
print(f"Very Greedy value: {value_VG}")
results['very_greedy'][func_name] = {
        'value': value_VG,
        'weight': weight_VG,
        'bitstring': bitstring_VG,
        'ratio_optim': value_VG / value_opt
    }


#%% ===================== LOADING JOBS RESULTS =====================

# Extract jobs
if not fake_hardware:
    print("Extracting jobs.")
    jobs = service.jobs(limit=300,
                        backend_name='ibm_quebec',
                        session_id=workload_id)
else:
    jobs = dict_jobs

# Initialize dictionary to store results
dict_jobs_results = {}

# Wrap the 'jobs' iterable with 'tqdm' for a progress bar
if not fake_hardware:
    for idx, job in enumerate(tqdm(jobs, desc="Loading Jobs", unit="job")):

        job_id = job.job_id()
        params = tuple(dict_jobs[job_id]['params'])
        dict_jobs_results[params] = {}  # Nested dict

        if job.done():
            if not fake_hardware:
                job_id = job.job_id()
                result = job.result()[0]
                counts = result.data.meas.get_counts()
            else:
                job_id = job
                counts = dict_jobs[job_id]['counts']
                params = dict_jobs[job_id]['params']

            # Find best solution -- If all the counts are one, will return the first bitstring
            bitstring = max(counts, key=counts.get)

            # Compute the value and weight for the bitstring
            value = get_value(bitstring, v, bit_mapping)
            weight = get_weight(bitstring, w, bit_mapping)

            # Save results to dictionary
            dict_jobs_results[params]['value'] = value
            dict_jobs_results[params]['weight'] = weight

            # only save counts for small computation to reduce space usage
            if n_units <= 50:
                dict_jobs_results[params]['counts'] = counts

            else: # only save the list of all bitstrings
                dict_jobs_results[params]['bitstrings'] = list(counts.keys())
                
        else:
            dict_jobs_results[params]['value'] = 0
            dict_jobs_results[params]['weight'] = 0


# Post-Process for Fake Hardware
if fake_hardware:

    for job_id, result in dict_jobs.items():
        params = tuple(result['params'])
        counts = result['counts']
        dict_jobs_results[params] = {}  # Nested dict

        # Find best solution -- If all the counts are one, will return the first bitstring
        bitstring = max(counts, key=counts.get)
        print('Bitstring:', bitstring)

        # Compute the value and weight for the bitstring
        value = get_value(bitstring, v, bit_mapping)
        weight = get_weight(bitstring, w, bit_mapping)

        # Save results to dictionary
        dict_jobs_results[params]['value'] = int(value)
        dict_jobs_results[params]['weight'] = int(weight)
        # print('weight', weight)

        # only save counts for small computation to reduce space usage
        if n_units <= 30:
            dict_jobs_results[params]['counts'] = counts
        else: # only save the list of all bitstrings
            dict_jobs_results[params]['bitstrings'] = list(counts.keys())


# Convert tuple keys to strings and ensure all values are JSON serializable
dict_jobs_results = {
    str(k): {sub_k: convert_to_serializable(sub_v) for sub_k, sub_v in v.items()}
    for k, v in dict_jobs_results.items()}

# Save to a JSON file
with open(f"{PATH_RUNS}/{exec_name}/dict_jobs_results.json", "w") as file:
    json.dump(dict_jobs_results, file, indent=4)
print(f"Saved JOBS results to {exec_name}/dict_jobs_results.json.")



#%% ===================== ANALYSING JOBS RESULTS =====================

# Open and read the JSON file
with open(f"{PATH_RUNS}/{exec_name}/dict_jobs_results.json") as file:
    dict_jobs_results = json.load(file)

# Convert back string keys to tuple
dict_jobs_results = {eval(key): value for key, value in dict_jobs_results.items()}

list_best_weight = []
list_best_params = []
best_value_quantum = -np.inf
dict_HM = {} # heat map dictionary

for idx, params in enumerate(dict_jobs_results):
    value = dict_jobs_results[params]['value']
    weight = dict_jobs_results[params].get('weight')
    # print('weight', weight)

    dict_HM[params] = value

    # Make sure the solution is valid
    if (weight != None) and (weight <= c):

        if value == best_value_quantum:
            list_best_weight.append(weight)
            list_best_params.append(params)
        
        if value > best_value_quantum: # update if we find a better solution
            list_best_weight = [weight]
            list_best_params = [params]

            best_value_quantum = value
            print('New best value', best_value_quantum)
            best_weight = weight
            best_params = params

# Counts from the best parameters
optimal_counts = dict_jobs_results[best_params].get('counts')

# Store results in dict
results['copula'][func_name] = {
    'ratio_gurobi': best_value_quantum / value_opt,
    'beta_opt': best_params[0],
    'gamma_opt': best_params[1],
    'value': best_value_quantum,
    'weight': best_weight,
    'bitstring': None,
    'ratio_optim': best_value_quantum / value_opt,
    }


#%% ===================== VISUALIZING RESULTS =====================

beta_list = sorted(set(job['params'][0] for job in dict_jobs.values()))
gamma_list = sorted(set(job['params'][1] for job in dict_jobs.values()))

plot_heatmap(dict_HM, 
             beta_list,
             gamma_list,
             value_opt,
             cmap='plasma',
             vmax=value_opt)

if optimal_counts:
    plot_custom_histogram(optimal_counts,
                          max_bitstrings=1000,
                          remove_xticks=True,
                          display_text=False)

    print(f"Nb counts: {len(optimal_counts)}/{2**n_units}\
        ({len(optimal_counts)/2**n_units*100:.2e}%)")
else:
    print("No counts were saved.")

# Results Histogram from HeatMap
plot_method_comparison(results, value_opt, methods=None, labels=None, title=None)


#%%
# Get all bitstrings from optimal heatmap parameters
try:
    bitstring_in_counts = dict_jobs_results[best_params]['bitstrings']
except:
    bitstring_in_counts = set(dict_jobs_results[best_params]['counts'].keys())

value, bitstring, weight = solve_knapsack(w, v, c, bitstring_in_counts)
print(f"Updated New Best Results: {value}")

# Update results to dict
results['copula_1'] = {}
results['copula_1'][func_name] = {
    'ratio_gurobi': value / value_opt,
    'beta_opt': best_params[0],
    'gamma_opt': best_params[1],
    'value': value,
    'weight': weight,
    'bitstring': bitstring,
    'ratio_optim': best_value_quantum / value_opt,
    }

# BruteForce all bitstring obtain by the QPU
unique_bitstring = extract_unique_bitstrings(dict_jobs_results)

value, bitstring, weight = solve_knapsack(w, v, c, unique_bitstring)
print(f"Updated New Best Results (ALL BITSTRING): {value}")

# Update results to dict
results['copula_2'] = {}
results['copula_2'][func_name] = {
    'ratio_gurobi': value / value_opt,
    'beta_opt': best_params[0],
    'gamma_opt': best_params[1],
    'value': value,
    'weight': weight,
    'bitstring': bitstring,
    'ratio_optim': best_value_quantum / value_opt,
    }

plot_method_comparison(
    results, value_opt,
    methods=['lazy_greedy', 'very_greedy', 'copula', 'copula_1', 'copula_2'],
    labels=['LG', 'VG', r'$QKP_{COP}$', r'$QKP_{COP}$ 2', r'$QKP_{COP}$ 3'],
    title=f"Performance Comparison Across Distributions with {n_units} items")


#%% =====================  Plot histogram values distribution  =====================

# Counts from the best parameters
if optimal_counts:
    bitstring_values_dist = {sum_values(sample_i, v): count
                        for sample_i, count in optimal_counts.items()
                        # if sum_weight(sample_i, w) <= c
                        }
    
    plot_histogram_with_vlines(bitstring_values_dist,
                               -value_opt,
                               log=False,
                               bins_width=200)

# Counts from all parameters combined
all_bitstring_values_dist = {sum_values(sample_i, v): 1
                            for sample_i in unique_bitstring
                            # if sum_weight(sample_i, w) <= c
                            }

plot_histogram_with_vlines(all_bitstring_values_dist,
                            -value_opt,
                            log=False,
                            bins_width=200)

# Try to plot all parameters to see the impact of (gamma, beta) on values distribution
list_bin_height = []
for params, job in dict_jobs_results.items():
    print(f"Parameters: {params}")
    counts = job['counts']
    bitstring_values_dist = {sum_values(sample_i, v): count
                    for sample_i, count in counts.items()
                    # if sum_weight(sample_i, w) <= c
                    }
    plot_histogram_with_vlines(bitstring_values_dist,
                                -value_opt,
                                log=False,
                                bins_width=100,
                                return_bin_height=False)
    
    # list_bin_height.append(bin_height)

# # Get the best and worst bitstring distribution
# arr_bin_height = np.array(list_bin_height)
# idx_max_bin_height = np.argmax(arr_bin_height)
# idx_min_bin_height = np.argmin(arr_bin_height)

# for idx, job in enumerate(dict_jobs_results.items()):

#     # Best case
#     if idx == idx_max_bin_height:
#         print(idx)
#         params = job[0]
#         counts = job[1]['counts']
#         bitstring_values_dist = {sum_values(sample_i, v): count
#                         for sample_i, count in counts.items()
#                         # if sum_weight(sample_i, w) <= c
#                         }
#         bin_height = plot_histogram_with_vlines(bitstring_values_dist,
#                                     -value_opt,
#                                     log=False,
#                                     bins_width=1000,
#                                     return_bin_height=False)

#     # Worst case
#     if idx == idx_min_bin_height:
#         print(idx)
#         params = job[0]
#         counts = job[1]['counts']
#         bitstring_values_dist = {sum_values(sample_i, v): count
#                         for sample_i, count in counts.items()
#                         # if sum_weight(sample_i, w) <= c
#                         }
#         bin_height = plot_histogram_with_vlines(bitstring_values_dist,
#                                     -value_opt,
#                                     log=False,
#                                     bins_width=1000,
#                                     return_bin_height=False)



# %% Compute Hamming distance with the Gurobi solution
hamming_dist = get_hamming_distance(bitstring, result_gurobi['bitstring'])
print(f"Hamming distance with optimal solution: {hamming_dist}")
print(f"Copula 3: {bitstring}")
print(f"Gurobi: {result_gurobi['bitstring']}")

# %%
