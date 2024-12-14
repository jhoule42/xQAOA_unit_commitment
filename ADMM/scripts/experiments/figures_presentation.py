#%%

import pickle
import logging
import numpy as np
from docplex.mp.model import Model

from qiskit.primitives import Sampler
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import MinimumEigenOptimizer, ADMMOptimizer, ADMMParameters
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeAuckland
from qiskit_ibm_runtime import SamplerV2, EstimatorV2

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

from UC.scripts.utils.models import *
from UC.scripts.utils.utils import *
from UC.scripts.utils.visualize import *
from UC.scripts.solvers.classical_solver_UC import *
from UC.scripts.solvers.admm_optimizer import ADMMParameters, ADMMOptimizer
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz



# %%

L = 100  # Total power demand
n_units = 4
lambda1 = 1000

A, B, C, p_min, p_max = generate_units(N=n_units)
param_exec = {"L": L,
              "n_units":n_units,
              "A": A,
              "B": B,
              "C": C,
              "p_min": p_min,
              "p_max": p_max}

# Suppress scientific notation, limit to 2 decimal places
np.set_printoptions(suppress=True, precision=2)

# Generate quadratic program
qp_UC = create_uc_model(A, B, C, L, p_min, p_max)
print(f"Linear:{qp_UC.objective.linear.to_array()}")
print(f"Quad:\n{qp_UC.objective.quadratic.to_array()}")


#%% ======================= Classical Solver Gurobi =======================
print("\nGurobi Solver")
bitstring_gurobi, power_distrib_gurobi, cost_gurobi, runtime_gurobi = classical_unit_commitment_qp(A, B, C, L, p_min, p_max)
result_gurobi = [bitstring_gurobi, power_distrib_gurobi, cost_gurobi, runtime_gurobi]
print(f"Bitstring: {bitstring_gurobi}")
print(f"Power distribution: {power_distrib_gurobi}")
print(f"Total Power Load: {np.sum(power_distrib_gurobi):.1f} | {L}")
print(f"Cost: {cost_gurobi:.2f}")


#%%

visualize_optimal_power_distribution(param_exec,
                                     Power_Distribution = power_distrib_gurobi)



# %%
