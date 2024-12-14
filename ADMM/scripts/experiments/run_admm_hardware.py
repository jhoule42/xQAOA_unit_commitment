""" Code to run the ADMM algorithm om IBM's hardware."""
#%%
import matplotlib.pyplot as plt

from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import Session, Options
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

#%% ============= ANALYSER JOB OUTPUT ==============

job_id = "cwe0a8h40e00008888d0"
job = service.job(job_id)

exp_val_list = job.result()[0].data.evs
print(exp_val_list)

job_id = "cwe0dhp0r6b0008p13rg"
job = service.job(job_id)

exp_val_list = job.result()[0].data.evs
print(exp_val_list)

#%%

# # qc should normaly be optimised for the quantum computer used
# # here we only use the least busy one
# service = QiskitRuntimeService(channel='ibm_quantum')
# backend = service.least_busy(operational=True, simulator=False)

# # Import quantum circuit
# ansatz = ansatz

# # Transpiling the quantum circuit
# pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
# isa_ansatz = pm.run(ansatz)
# isa_observable = observable_2.apply_layout(layout = isa_ansatz.layout)


# candidate_circuit = pm.run(circuit) # start transpiling
# candidate_circuit.draw('mpl', fold=False, idle_wires=False)


# # Run minimisation process
# objective_func_vals = [] # Global variable
# with Session(backend=backend) as session:

#     estimator = Estimator(mode=session)
#     estimator.options.default_shots = 1000

#     # Set simple error suppression/mitigation options
#     estimator.options.dynamical_decoupling.enable = True
#     estimator.options.dynamical_decoupling.sequence_type = "XY4"
#     estimator.options.twirling.enable_gates = True
#     estimator.options.twirling.num_randomizations = "auto"

#     result = minimize(
#         cost_func_estimator,
#         init_params,
#         args=(candidate_circuit, cost_hamiltonian, estimator),
#         method="COBYLA",
#         tol=1e-2)

#     print(result) # solution of the optimisation

# # plotting cost value convergence
# plt.figure(figsize=(12, 6))
# plt.plot(objective_func_vals)
# plt.xlabel("Iteration")
# plt.ylabel("Cost")
# plt.show()


# # Once the optimal parameters have been found, we assign these parameters
# # and sample the final distribution
# optimized_circuit = candidate_circuit.assign_parameters(result.x)
# optimized_circuit.draw('mpl', fold=False, idle_wires=False)



# # Run Sample
# sampler = Sampler(mode=backend)
# sampler.options.default_shots = 1000

# # Set simple error suppression/mitigation options
# sampler.options.dynamical_decoupling.enable = True
# sampler.options.dynamical_decoupling.sequence_type = "XY4"
# sampler.options.twirling.enable_gates = True
# sampler.options.twirling.num_randomizations = "auto"

# pub = (optimized_circuit, )
# job = sampler.run([pub], shots=int(1e4))


# counts_int = job.result()[0].data.meas.get_int_counts()
# counts_bin = job.result()[0].data.meas.get_counts()
# shots = sum(counts_int.values())
# final_distribution_int = {key: val/shots for key, val in counts_int.items()}
# final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
# print(final_distribution_int)
