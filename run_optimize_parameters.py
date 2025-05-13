#%% QAOA Parameter Optimization Script for Knapsack Problem (KP)
# This script constructs and optimizes a QAOA circuit for the KP using a copula-based mixer.
# It supports both hardware execution (IBM Quantum backend) and simulation (MPS backend).
# Optional light-cone transpilation is available for circuit reduction.
# The script saves results, plots optimization progress, and compares quantum/classical solutions.

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

sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

from xQAOA.scripts.utils.kp_utils import *
from xQAOA.scripts.solvers.qkp_solver import *
from xQAOA.scripts.utils.visualize import *
from ADMM.scripts.solvers.classical_solver_UC import gurobi_knapsack_solver


#%% ============================= BACKEND CONNECTION =============================
service = QiskitRuntimeService(channel="ibm_quantum",
                               instance='pinq-quebec-hub/universit-de-cal/main')
backend = service.backend('ibm_quebec')
print("Backend Connected.")

#%% ============================= Set Parameters =============================
n = 40 # number of qubits
shots = 100000 # number of shots
p = 1 # number of layers for the QAOA circuit
load_factor = 0.8  # if set to None; random load between [0.25-0.75] max weight
light_cone = True
full_circuit_optimization = True
func_distribution = generate_profit_spanner
transpilation_level = 3

k_range = [15]  # Simplified from np.arange(10, 24, 1)
theta_range = [-1]  # Simplified from [0, -0.5, -1]
bit_mapping = 'regular' # choose 'inverse' to solve UC
init_params = [np.pi, np.pi / 2] * p  # (gamma, beta) pairs initialization

# Generate values and weights for the current distributioné
v, w = func_distribution(n)
c = np.ceil(load_factor * sum(w)).astype(int)

PATH_RUNS = "/Users/julien-pierrehoule/Documents/Stage/T3/Code/xQAOA/runs/hardware"
folder_name = f"KP_N{n}_optimized_{func_distribution.__name__[9:]}_load-{load_factor}_p{p}"

# Save the parameters for the execution to a dictionary
dict_params = {}
dict_params['execution_type'] = "hardware"
dict_params['n_units'] = n
dict_params['load_factor'] = load_factor
dict_params['distribution'] = func_distribution.__name__
dict_params['p'] = p
dict_params['theta_range'] = theta_range
dict_params['k_range'] = k_range
dict_params['bit_mapping'] = bit_mapping
dict_params['transpilation_level'] = transpilation_level
dict_params['shots'] = shots
dict_params['v'] = [int(i) for i in list(v)]
dict_params['w'] = [int(i) for i in list(w)]
dict_params['c'] = int(c)

os.makedirs(f"{PATH_RUNS}/{folder_name}", exist_ok=True)
print(f"Folder created: {folder_name}")


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
print(f"Logical circuit depth: {qaoa_ansatz.decompose().depth()}")


#%% ======================= Initialize the Estimator ========================

# Implementing noise model takes a lot of time to run
nm = NoiseModel.from_backend(backend)
# nm = None
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

    pbar = tqdm(total=len(hamiltonian.paulis), desc="Light cone computation")
    for i, pauli_term in enumerate(hamiltonian.paulis):
        pauli_str = pauli_term.to_label()
        active_indices = [j for j, p in enumerate(pauli_str) if p != 'I']
        if not active_indices:
            light_cone_list.append(None)
            pbar.update(1)
            continue

        t0 = time.perf_counter() # timer for the progress bar
        lc_pass = LightCone(bit_terms="Z", indices=active_indices)
        reduced_dag = lc_pass.run(circuit_to_dag(ansatz))
        reduced_circuit = dag_to_circuit(reduced_dag)
        elapsed = time.perf_counter() - t0

        light_cone_list.append(reduced_circuit)
        pbar.set_postfix(last_time=f"{elapsed:.6f} s")
        pbar.update(1)
    pbar.close()

    return light_cone_list



def cost_func_lightcone(params_values, ansatz, hamiltonian, estimator, light_cone_list):
    """Compute total expectation using precomputed light-cone-reduced circuits with EstimatorV2."""

    total = float(hamiltonian.coeffs[-1].real)
    param_dict = dict(zip(ansatz.parameters, params_values))

    for i, pauli_term in enumerate(hamiltonian.paulis):
        coeff = hamiltonian.coeffs[i].real
        pauli_str = pauli_term.to_label()

        if light_cone_list[i] is None:
            continue  # skip identity-only terms (handled in the constant term)

        reduced_circuit = light_cone_list[i]
        reduced_params = [param_dict[p] for p in reduced_circuit.parameters if p in param_dict]
        reduced_pauli = SparsePauliOp(pauli_str)

        job = estimator.run([(reduced_circuit, reduced_pauli, reduced_params)])
        result = job.result()[0]
        ev = result.data.evs
        total += coeff * ev

    objective_func_vals_lc.append(total)
    return total


#%% ==================== RUN THE LIGHT CONE OPTIMIZATION =====================

if light_cone:
    print("Running the light cone optimization...")

    # Transpile & layout
    transpiled = transpile(qaoa_ansatz, basis_gates=basis_gates, optimization_level=3)
    mapped_H = cost_hamiltonian.apply_layout(transpiled.layout)

    # Precompute light-cone circuits
    lc_list = compute_light_cone_circuits(transpiled, mapped_H)

    # Optimize
    result_lc = minimize(
        cost_func_lightcone,
        init_params,
        args=(transpiled_circuit, mapped_H, estimator_mps, lc_list),
        method="COBYLA",
        bounds=[(0, np.pi), (0, 2*np.pi)] * p,
        tol=1e-6,
        options={"maxiter":150, "disp":True},
        callback=lambda xk: print(f"Iteration {len(objective_func_vals_lc)}: Current params = {xk}"),
    )
    print("Optimal cost (light-cone):", result_lc.fun)


#%% ======================= RUN THE OPTIMIZATION ======================== 

def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    """ The cost function to be minimized."""

    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs
    objective_func_vals.append(cost)

    return -cost


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
with open(f'{PATH_RUNS}/{folder_name}/parameters.json', 'w') as file:
    json.dump(dict_params, file, indent=4)
print('Execution parameters saved to file.')

#%% ======================== PLOT THE COST FUNCTION ========================
plt.figure(figsize=(8, 5))
plt.plot(objective_func_vals, marker='o', linestyle='-', label='Regular')
plt.plot(objective_func_vals_lc, marker='o', linestyle='-', label='Light Cone')
plt.xlabel("Iteration")
plt.ylabel("Cost Function Value")
plt.title("Evolution of the Cost Function per Iteration")
plt.grid(True)
plt.legend()
plt.show()


# %% ======================== RUN THE SAMPLER on HARDWARE ========================

from qiskit_ibm_runtime import SamplerV2, Batch
from qiskit_ibm_runtime import QiskitRuntimeService

dict_jobs_id = {}
opt_circuit_reg = transpiled_circuit.assign_parameters(result_reg.x)
opt_circuit_reg.measure_active() 
print(f"Nb of 2Qbits gates depth: {opt_circuit_reg.depth(lambda x: (len(x.qubits)>=2))}")

with Batch(backend=backend, max_time='3h') as batch:
    sampler = SamplerV2(mode=batch)
    print('Sending Batch Job.')

    workload_id = batch.session_id

    # Set Error Mitigation
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates = True # enable Pauli Twirling
    sampler.options.twirling.enable_measure = True

    job_reg = sampler.run([(opt_circuit_reg,)], shots=shots)

print('Done submitting jobs.')



# %% =================== RUN THE SAMPLER (simulation) ===================

print("Running the sampler.")
# opt_circuit_lc = transpiled_circuit.assign_parameters(result_lc.x)
# opt_circuit_lc.measure_all()  # add measurement

opt_circuit_reg = transpiled_circuit.assign_parameters(result_reg.x)
opt_circuit_reg.measure_active()  # add measurement

# job_lc = sampler_mps.run([(opt_circuit_lc,)], shots=shots)
job_reg = sampler_mps.run([(opt_circuit_reg,)], shots=shots)

# counts_int_lc = job_lc.result()[0].data.meas.get_int_counts()
# counts_bin_lc = job_lc.result()[0].data.meas.get_counts()

counts_int_reg = job_reg.result()[0].data.meas.get_int_counts()
counts_bin_reg = job_reg.result()[0].data.meas.get_counts()

#%% ======================== POST-PROCESS RESULTS ========================

# Gurobi Solver Knapsack
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
dict_bit_values = convert_bitstring_to_values(counts_bin_reg, v, w, c,
                                              filter_invalid_solutions=filter_sols)

# Sort the dictionary by value
sorted_dict = dict(sorted(dict_bit_values.items(), key=lambda item: item[0], reverse=True))

# Get the best value
best_value = max(sorted_dict.keys())
best_count = sorted_dict[best_value]
print(f"Best value: {best_value}")
print(f"Best count: {best_count}")

# Compute the probability of sucess and the approximate ratio
aprox_ratio = compute_approximate_ratio(dict_bit_values, value_opt)


# %%
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
    "Random": dict_bit_values_random,
    f"Copula p={p}": dict_bit_values,
}
# Custom colors
colors = {
    "Random": "orange",
    f"Copula p={p}": "steelblue",
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
    # nb_bins=len(dict_bit_values),
    log=False,
    annotations=annotations,
    figsize=(8, 6),
)

# %% ======================= PLOT THE COST FUNCTION ========================

# plot the parameters optimization
iterations = np.arange(len(result_reg.x)//2) + 1
plt.figure(figsize=(8, 5))
plt.plot(iterations, result_reg.x[0::2], marker='o', linestyle='-', label='Gamma')
plt.plot(iterations, result_reg.x[1::2], marker='o', linestyle='-', label='Beta')
plt.xlabel("Iteration")
plt.ylabel("Parameter Value")
plt.legend()
plt.grid(True)
plt.show()

# %%


