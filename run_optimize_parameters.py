#%% QAOA Parameter Optimization Script for Knapsack Problem (KP)
# This script constructs and optimizes a QAOA circuit for the KP using a copula-based mixer.
# It supports both hardware execution (IBM Quantum backend) and simulation (MPS backend).
# Optional light-cone transpilation is available for circuit reduction.

import os, sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import minimize
from tqdm import tqdm

from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.passes import LightCone, RemoveBarriers
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import QAOAAnsatz
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_aer.noise import NoiseModel
# from qiskit_algorithms.optimizers import SPSA

#sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")
sys.path.append("..")
from scripts.utils.kp_utils import *
from qkp_solver import *
from scripts.utils.visualize import *
from ADMM.scripts.solvers.classical_solver_UC import gurobi_knapsack_solver


#%% ============================= BACKEND CONNECTION =============================
service = QiskitRuntimeService(name='enablement-work')
backend = service.backend('ibm_torino')
print("Backend Connected.")

#%% ============================= Set Parameters =============================
n = 104 # number of qubits
shots = 100000 # number of shots
p = 1 # number of layers for the QAOA circuit
load_factor = 0.8  # if set to None; random load between [0.25-0.75] max weight
light_cone = True
full_circuit_optimization = False
func_distribution = generate_inversely_strongly_correlated
# func_distribution = generate_profit
transpilation_level = 2

k_range = 5  # Simplified from np.arange(10, 24, 1)
theta_range = [-1]  # Simplified from [0, -0.5, -1]
bit_mapping = 'regular' # choose 'inverse' to solve UC

# Generate values and weights for the current distributioné
v, w = func_distribution(n)
c = np.ceil(load_factor * sum(w)).astype(int)

#PATH_RUNS = "/Users/julien-pierrehoule/Documents/Stage/T3/Code/xQAOA/runs/hardware"
#folder_name = f"KP_N{n}_optimized_{func_distribution.__name__[9:]}_load-{load_factor}_p{p}_k{k_range}"

# Save the parameters for the execution to a dictionary
# dict_params = {}
# dict_params['execution_type'] = "hardware"
# dict_params['n_units'] = n
# dict_params['load_factor'] = load_factor
# dict_params['distribution'] = func_distribution.__name__
# dict_params['p'] = p
# dict_params['theta_range'] = theta_range
# dict_params['k_range'] = [k_range]
# dict_params['bit_mapping'] = bit_mapping
# dict_params['transpilation_level'] = transpilation_level
# dict_params['shots'] = shots
# dict_params['v'] = [int(i) for i in list(v)]
# dict_params['w'] = [int(i) for i in list(w)]
# dict_params['c'] = int(c)

# os.makedirs(f"{PATH_RUNS}/{folder_name}", exist_ok=True)
# print(f"Folder created: {folder_name}")


#%% ======================== Construct the QAOA circuit ========================

def cost_hamiltonian_paulis_op(n, v):
    """ Construct the cost Hamiltonian as a SparsePauliOp."""
    # Construct the cost Hamiltonian: H_C = sum_i (v_i / 2) * (I - Z_i)
    pauli_terms = []
    coeffs = []

    for i in range(n):
        # Z_i term
        z_string = ['I'] * n
        z_string[i] = 'Z'
        pauli_terms.append(''.join(z_string))
        coeffs.append(-v[i] / 2)  # coefficient for -Z_i

    # Add the identity part: sum_i v_i / 2
    pauli_terms.append('I' * n)
    coeffs.append(sum(v) / 2)

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))

cost_hamiltonian = cost_hamiltonian_paulis_op(n, v)

# Create QAOA ansatz using this cost Hamiltonian
gamma =  ParameterVector('γ', p)
beta = ParameterVector('β', p)
optimizer_C = QKPOptimizer(v, w, c,
                            mixer='copula_2',
                            generate_jobs=True,
                            run_hardware=True,
                            backend=None,
                            pass_manager=None,
                            p=p,)

p_dist = optimizer_C.logistic_bias(k=15)
initial_state_qc = optimizer_C.initial_state_preparation(QuantumCircuit(n), p_dist)

cost_qc = QuantumCircuit(n)
mixer_qc = QuantumCircuit(n)

cost_layer = optimizer_C.generate_cost_unitary(gamma=gamma[0])
mixer_layer = optimizer_C.ring_copula_mixer(QuantumCircuit(n), p_dist,theta=-1, beta=beta[0])
cost_qc.compose(cost_layer, inplace=True)
mixer_qc.compose(mixer_layer, inplace=True)

# Construct the QAOA ansatz with both gamma and beta vector elements
qaoa_ansatz = QAOAAnsatz(cost_operator=cost_layer,
                         initial_state=initial_state_qc,
                         mixer_operator=mixer_qc,
                         reps=p)

# Remove barriers from the circuit (necessary for light cone transpilation)
qaoa_ansatz = RemoveBarriers()(qaoa_ansatz.decompose())
print(f"Logical circuit depth: {qaoa_ansatz.decompose().depth(lambda x: len(x.qubits)==2)}")


#%% ======================= Initialize the Estimator ========================

# Implementing noise model takes a lot of time to run
# nm = NoiseModel.from_backend(backend)
nm = None
coupling_map = backend.configuration().coupling_map
basis_gates = backend.basis_gates

# initialize noisy simulator
estimator_mps = EstimatorV2(
    options={"backend_options": {"method": "matrix_product_state",
                                 "noise_model": nm,
                                 "coupling_map":coupling_map,
                                 },
                               })

sampler_mps = SamplerV2(
    options={"backend_options": {"method": "matrix_product_state",
                                 "noise_model": nm,
                                 "coupling_map":coupling_map,
                                 },
                                 })

print("Transpiling the circuit...")
transpiled_circuit = transpile(qaoa_ansatz, backend=backend,
                               optimization_level=transpilation_level)

print(f"(Transpiled) Circuit depth: {transpiled_circuit.depth()}")
print(f"(Transpiled) 2Q gates depth: {transpiled_circuit.depth(lambda x: (len(x.qubits)>=2))}")
print(f"(Transpiled) Total 2Q gates: {transpiled_circuit.num_nonlocal_gates()}")


#%% ======================= RUN THE OPTIMIZATION ======================== 

# Initialize
objective_func_vals_lc = []
list_reduced_circuit = []
params = list(qaoa_ansatz.parameters)

def compute_light_cone_circuits(ansatz, hamiltonian):
    """
    Compute the light cone reduced circuits for each Pauli term in the Hamiltonian.
    For identity terms (no active qubits), None is stored.
    Returns a list of reduced circuits.
    """
    light_cone_list = []
    reduced_paulis = []
    num_qubits = len(hamiltonian.paulis[0])

    pbar = tqdm(total=len(hamiltonian.paulis), desc="Light cone computation")
    for i, pauli_term in enumerate(hamiltonian.paulis):

        pauli_str = pauli_term.to_label()
        active_indices = [(num_qubits-1-i) for i, p in enumerate(pauli_str) if p != "I"]

        if not active_indices:
            light_cone_list.append(None)
            reduced_paulis.append(None)
            pbar.update(1)
            continue

        t0 = time.perf_counter() # timer for the progress bar
        lc_pass = LightCone(bit_terms="Z", indices=active_indices)
        reduced_dag = lc_pass.run(circuit_to_dag(ansatz))
        reduced_circuit = dag_to_circuit(reduced_dag)
        elapsed = time.perf_counter() - t0

        light_cone_list.append(reduced_circuit)
        reduced_paulis.append(pauli_term)

        pbar.set_postfix(last_time=f"{elapsed:.6f} s")
        pbar.update(1)
    pbar.close()

    return light_cone_list, reduced_paulis


def cost_func_lightcone(params_values, ansatz, hamiltonian, estimator, light_cone_data):
    """Compute total expectation using precomputed light-cone-reduced circuits with EstimatorV2."""

    light_cone_list, reduced_paulis = light_cone_data
    total = 0.0
    param_dict = dict(zip(ansatz.parameters, params_values))

    for i, pauli_term in enumerate(hamiltonian.paulis):
        coeff = hamiltonian.coeffs[i].real
        pauli_str = pauli_term.to_label()

        if light_cone_list[i] is None:
            # Handle identity term
            total += coeff
            continue

        reduced_circuit = light_cone_list[i]
        reduced_params = [param_dict[p] for p in reduced_circuit.parameters if p in param_dict]
        # reduced_pauli = SparsePauliOp(pauli_str)

        # Run estimator with the correct Pauli operator
        job = estimator.run([(reduced_circuit, reduced_paulis[i], reduced_params)])
        result = job.result()[0]
        ev = result.data.evs
        total += coeff * ev

    objective_func_vals_lc.append(total)
    return -total


#%% 
init_params = [np.pi/8, np.pi / 8] * p  # (gamma, beta) pairs initialization

if light_cone:
    print("Running the light cone optimization...")
    # Use the already existing transpiled_circuit
    mapped_H_for_lc = cost_hamiltonian.apply_layout(transpiled_circuit.layout)
    lc_data = compute_light_cone_circuits(transpiled_circuit, mapped_H_for_lc) # Pass the same circuit

    # Optimize
    result_lc = minimize(
        cost_func_lightcone,
        init_params,
        args=(transpiled_circuit, mapped_H_for_lc, estimator_mps, lc_data),
        method="COBYLA",
        bounds=[(0, np.pi), (0, 2*np.pi)] * p,
        tol=1e-6,
        options={"maxiter":1000, "disp":True},
        # rhobeg=0.1, # parameter to play with
        callback=lambda xk: print(f"Iteration {len(objective_func_vals_lc)}: Current params = {xk}"),
    )

    # spsa = SPSA(maxiter=300) # Set the maximum number of iterations

    # result_lc = spsa.minimize(
    #     fun=cost_func_lightcone,
    #     x0=init_params,
    #     bounds=[(0, np.pi), (0, 2*np.pi)] * p,
    # )
    # print("Optimal cost (light-cone):", result_lc.fun)

else:
    print("Light cone optimization not performed.")
    result_lc = None

 #%% ========================= RUN THE OPTIMIZATION ========================= 

def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    """ The cost function to be minimized."""

    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs
    objective_func_vals.append(cost)

    return -cost, 


if full_circuit_optimization:
    print("Running the full circuit optimization.")
    objective_func_vals = [] # Global variable

    def callback(xk):
        print(f"Iteration {len(objective_func_vals)}: Current params = {xk}")

    result_reg = minimize(
        cost_func_estimator,
        init_params,
        args=(transpiled_circuit, cost_hamiltonian, estimator_mps),
        method="COBYLA",
        bounds=[(0, np.pi), (0, 2*np.pi)] * (len(init_params)//2),
        tol=1e-6,
        options={"maxiter": 150,  "disp": True},  
        callback=callback,
    )
    print("Optimal cost (regular):", result_reg.fun)

    dict_params['optimized_params'] = [float(i) for i in result_reg.x]
    dict_params['cost_function'] = float(result_reg.fun)

# Save parameter and optimization results to file
# with open(f'{PATH_RUNS}/{folder_name}/parameters.json', 'w') as file:
#     json.dump(dict_params, file, indent=4)
# print('\nExecution parameters saved to file.')


#%% ======================== PLOT THE COST FUNCTION ========================
plt.figure(figsize=(8, 5))
# plt.plot(objective_func_vals, marker='o', linestyle='-', label='Regular')
plt.plot(objective_func_vals_lc, marker='o', linestyle='-', label='Light Cone')
print("max cost-fn value:", max(objective_func_vals_lc))
plt.xlabel("Iteration")
plt.ylabel("Cost Function Value")
plt.title("Evolution of the Cost Function per Iteration")
plt.grid(True)
plt.legend()
plt.show()

# %% ======================== RUN THE SAMPLER on HARDWARE ========================

# from qiskit_ibm_runtime import SamplerV2, Batch
# from qiskit_ibm_runtime import QiskitRuntimeService

# dict_jobs_id = {}
# opt_circuit_reg = transpiled_circuit.assign_parameters(result_lc.x)
# opt_circuit_reg.measure_active() 
# print(f"Nb of 2 qbits gates depth: {opt_circuit_reg.depth(lambda x: (len(x.qubits)>=2))}")

# with Batch(backend=backend, max_time='3h') as batch:
#     sampler = SamplerV2(mode=batch)
#     print('Sending Batch Job.')

#     workload_id = batch.session_id

#     # Set Error Mitigation
#     sampler.options.dynamical_decoupling.enable = True
#     sampler.options.dynamical_decoupling.sequence_type = "XpXm"
#     sampler.options.twirling.enable_gates = True # enable Pauli Twirling
#     sampler.options.twirling.enable_measure = True

#     job_reg = sampler.run([(opt_circuit_reg,)], shots=shots)

# print('Done submitting jobs.')



# %% =================== RUN THE SAMPLER (simulation) ===================

# opt_params = np.random.uniform(0, 2*np.pi, size=2*p)

print("Running the sampler.")

if light_cone:
    opt_circuit_lc = transpiled_circuit.assign_parameters(result_lc.x)
    opt_circuit_lc.measure_active()  # add measurement
    job_lc = sampler_mps.run([(opt_circuit_lc,)], shots=shots)
    counts_int_lc = job_lc.result()[0].data.meas.get_int_counts()
    counts_bin_lc = job_lc.result()[0].data.meas.get_counts()

# if full_circuit_optimization:
#     opt_circuit_reg = transpiled_circuit.assign_parameters(result_reg.x)
#     opt_circuit_reg.measure_active()  # add measurement
#     job_reg = sampler_mps.run([(opt_circuit_reg,)], shots=shots)
#     counts_int_reg = job_reg.result()[0].data.meas.get_int_counts()
#     counts_bin_reg = job_reg.result()[0].data.meas.get_counts()


#%% ======================== POST-PROCESS RESULTS ========================

#Gurobi Solver Knapsack
result_gurobi = gurobi_knapsack_solver(v, w, c, verbose=False,
                                       time_limit=60,
                                       optimality_gap=1e-20,
                                       feasibility_tolerance=1e-9)
value_opt = result_gurobi['total_value']
print(f"Optimization Results Gurobi -- {n} items")
print(f"Bitstring: {result_gurobi['bitstring']}")


# %% ======================= POST-PROCESS RESULTS ========================

# Convert the counts to values
filter_sols = True

if light_cone:
    dict_bit_values_lc = convert_bitstring_to_values(counts_bin_lc, v, w, c,
                                            filter_invalid_solutions=filter_sols)

# if full_circuit_optimization:
#     dict_bit_values_full = convert_bitstring_to_values(counts_bin_reg, v, w, c,
#                                             filter_invalid_solutions=filter_sols)

# Sort the dictionary by value
sorted_dict = dict(sorted(dict_bit_values_lc.items(), key=lambda item: item[0], reverse=True))
best_value = max(sorted_dict.keys())
best_count = sorted_dict[best_value]
print(f"Best value: {best_value}")

# Compute the probability of sucess and the approximate ratio
aprox_ratio = compute_approximate_ratio(dict_bit_values_lc, value_opt)


# %%

print("\nGreedy Warm Start Solution")

def logistic_bias(v, w, c, k):
    """Creates a biased initial distribution using the logistic function."""
    r = np.array(v) / np.array(w)
    C = (sum(w) / c) - 1
    return 1 / (1 + C * np.exp(-k * (r - r.mean())))

# Calculate probabilities using logistic_bias
p_dist = logistic_bias(v, w, c, k=5)
r = v/w # Calculate r_i = v_i/w_i ratios

warm_start_counts = defaultdict(int)
shots = 100000

for _ in range(shots):
    bitstring = ''.join(str(int(np.random.random() < pi)) for pi in p_dist)
    warm_start_counts[bitstring] += 1

# Convert to value distribution
dict_bit_values_ws = convert_bitstring_to_values(warm_start_counts, v, w, c,
                                                   filter_invalid_solutions=filter_sols)

# Compute the probability of sucess
p_sucess_ws= probabilty_success(dict_bit_values_ws, value_opt)
aprox_ratio_ws = compute_approximate_ratio(dict_bit_values_ws, value_opt)

print("\nRandom Distribution Solution")

random_counts = defaultdict(int) # generate dict with default value of 0
for _ in range(shots):
    bitstring = ''.join(str(x) for x in np.random.randint(2, size=n))
    random_counts[bitstring] += 1

dict_bit_values_random = convert_bitstring_to_values(random_counts, v, w, c,
                                                     filter_invalid_solutions=filter_sols)

# Compute the probability of sucess
aprox_ratio_random = compute_approximate_ratio(dict_bit_values_random, value_opt)
print(f"Best value: {max(dict_bit_values_random.keys())}")

data_dict = {
f"Random: Apr={aprox_ratio_random:.2f}": dict_bit_values_random,
   f"WS: Apr={aprox_ratio_ws:.3f}": dict_bit_values_ws,
   f"Cop p=1: Apr={aprox_ratio:.3f}": dict_bit_values_lc,
}
# Custom colors
colors = {
    f"Random: Apr={aprox_ratio_random:.2f}": "orange",
    f"WS: Apr={aprox_ratio_ws:.3f}": 'black',
    f"Cop p=1: Apr={aprox_ratio:.3f}": "steelblue",
}    
# Annotations
annotations = {
    r"$\mathrm{Approximate\ Ratio}$": np.round(aprox_ratio, 2),
    "Ratio optimality (%)": np.round(best_value /value_opt *100, 2),
}
plot_multiple_distributions(
    data_dict=data_dict,
    min_cost=value_opt,
    colors=colors,
    nb_bins=2000,
    #nb_bins=len(dict_bit_values),
    log=False,
    annotations=annotations,
    figsize=(8, 6),
    title=f"Distribution:{func_distribution.__name__[9:]}  [k={k_range}]  Cap={load_factor*100}%"
)

# %% ======================= PLOT THE COST FUNCTION ========================
# plot the parameters optimization
iterations = np.arange(len(result_lc.x)//2) + 1
plt.figure(figsize=(8, 5))
plt.plot(iterations, result_lc.x[0::2], marker='o', linestyle='-', label='Gamma')
plt.plot(iterations, result_lc.x[1::2], marker='o', linestyle='-', label='Beta')
plt.xlabel("Iteration")
plt.ylabel("Parameter Value")
plt.legend()
plt.grid(True)
plt.show()

# %%

plt.figure(figsize=(8, 5))
for i in range(1,8):
    for j in range(1,8):
        objective_func_vals = [] 
        params = [np.pi/i, 2*np.pi /j] * p
        result_reg = minimize(
        cost_func_estimator,
        init_params,
        args=(transpiled_circuit, cost_hamiltonian, estimator_mps),
        method="COBYLA",
        bounds=[(0, np.pi), (0, 2*np.pi)] * (len(init_params)//2),
        tol=1e-6,
        options={"maxiter": 150,  "disp": True},  
        callback=None,
        )
    
    
    plt.plot([(i,j)], max((objective_func_vals_lc)),marker='x')
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function Value")
    plt.title("Evolution of the Cost Function per Iteration")
    plt.grid(True)
    plt.legend()
plt.show()


# %%
