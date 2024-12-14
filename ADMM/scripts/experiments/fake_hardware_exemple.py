""" Code use to test and debug the use of Fake Hardware Simulation using Qiksit 1.0 """
#%%
import numpy as np
from scipy.optimize import minimize

from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp

from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime import SamplerV2, EstimatorV2
from qiskit.visualization import plot_histogram

from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


# %% QAOA CIRCUIT

# backend = GenericBackendV2(num_qubits=2)
# backend = FakeManilaV2()
backend = FakeSherbrooke()

# Toy cost operator for testing
cost_operator = SparsePauliOp(['ZZ', 'II'], np.array([1.0, -0.5]))
p = 2 # circuit layer

# Create QAOA ansatz and optimize
circuit = QAOAAnsatz(cost_operator=cost_operator, reps=p)
pass_manager = generate_preset_pass_manager(3, backend=backend)
isa_circuit = pass_manager.run(circuit)

# Initial parameter setup
x0_params = np.empty(2*p, dtype=float)
x0_params[0::2] = np.linspace(0, 2*np.pi, p)  # Fill even indices with gammas
x0_params[1::2] = np.linspace(np.pi, 0, p)  # Fill odd indices with betas


def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    """Evaluate the cost function using the estimator to run QAOA."""

    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs

    return cost

#%%

with Session(backend=backend) as session:

    estimator = EstimatorV2(mode=session)
    results = minimize(cost_func_estimator,
                    x0=x0_params,
                    args=(isa_circuit, cost_operator, estimator),
                    method="COBYLA",
                    options={'maxiter': 10000, 'disp':True})
    
    print(results)
print('End of session')
isa_circuit.draw('mpl', fold=-1)


#%% Sampling the binary solution from the quantum circuit

def count_gate(qc):
    gate_count = {qubit : 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count


measured_circuit = isa_circuit.copy()
# measured_circuit.measure_all()
measured_circuit.measure_active()



exact_sampler = SamplerV2(backend)
isa_measured = pass_manager.run(measured_circuit)
isa_measured.draw('mpl', fold=-1)

#%%
pub = (isa_measured, results.x, 1000)
job = exact_sampler.run([pub])
result = job.result()
pub_result = result[0]
# counts = pub_result.data.meas.get_counts() # use this if measure_all() is used
counts = pub_result.data.measure.get_counts()

plot_histogram(counts)


# Get the binary solution with the highest count
x_sol = max(counts, key=counts.get)
print(f"Binary solution from QAOA: {x_sol}")
# %%
