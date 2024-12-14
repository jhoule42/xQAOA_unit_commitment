""" Code to run the ADMM algorithm."""
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

from UC.scripts.utils.utils import *
from UC.scripts.utils.models import *
from UC.scripts.utils.visualize import *
from UC.scripts.solvers.classical_solver_UC import *
from UC.scripts.solvers.admm_optimizer import ADMMParameters, ADMMOptimizer
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz


#%% ============================ Unit Commitment Model ============================
L = 200  # Total power demand
n_units = 6
lambda1 = 10000

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

# # Add quadratic terms
# qp_UC = cross_terms_matrix(qp_UC, lambda1, p_min, p_max, L)
# print(f"Linear:{qp_UC.objective.linear.to_array()}")
# print(f"Quad:\n{qp_UC.objective.quadratic.to_array()[4:, 4:]}")



#%% ======================= Classical Solver Gurobi =======================
print("\nGurobi Solver")
bitstring_gurobi, power_distrib_gurobi, cost_gurobi, runtime_gurobi = classical_unit_commitment_qp(A, B, C, L, p_min, p_max)
result_gurobi = [bitstring_gurobi, power_distrib_gurobi, cost_gurobi, runtime_gurobi]
print(f"Bitstring: {bitstring_gurobi}")
print(f"Power distribution: {power_distrib_gurobi}")
print(f"Total Power Load: {np.sum(power_distrib_gurobi):.1f} | {L}")
print(f"Cost: {cost_gurobi:.2f}")


#%% ========================= ADMM QUBO DIAGONAL TERMS =========================

# Set parameters
admm_params = ADMMParameters(rho_initial=1e10, beta=1000, factor_c=100, maxiter=100,
                             three_block=False, tol=1e-12, warm_start=True, p=3, vary_rho=0,
                             cross_terms=False)

# CLASSICAL SOLVER
admm_CS = ADMMOptimizer(params=admm_params, qubo_type='classical')
result_CS = admm_CS.solve(qp_UC)

x_sol = result_CS.x[:n_units] # extract binary string
power_distrib = result_CS.x[n_units:] # extract power distribution

print("\nADMM Classical QUBO")
print(f"Cost: {result_CS.fval:.1f}")
print(f"Bitstring: {x_sol}")
print(f"Power distribution: {power_distrib}")
print(f"Total Power Load: {np.sum(power_distrib):.1f} | {L}")


#%% ========================= ADMM QUBO CROSS TERMS =========================

# Set parameters
admm_params = ADMMParameters(rho_initial=1e12, beta=1000, factor_c=100, maxiter=200,
                             three_block=False, tol=1e-12, warm_start=True, p=3, vary_rho=0,
                             cross_terms=True)

# CLASSICAL SOLVER
admm_CS_CT = ADMMOptimizer(params=admm_params, qubo_type='classical')
result_CS_CT = admm_CS_CT.solve(qp_UC)

x_sol = result_CS_CT.x[:n_units] # extract binary string
power_distrib = result_CS_CT.x[n_units:] # extract power distribution

print("\nADMM Classical QUBO")
print(f"Cost: {result_CS_CT.fval:.1f}")
print(f"Bitstring: {x_sol}")
print(f"Power distribution: {power_distrib}")
print(f"Total Power Load: {np.sum(power_distrib):.1f} | {L}")



#%% ======================= Saving Results =======================
# # Save all relevant data
# run_description = """Run ADMM Simulation #1"""

# PATH_RESULTS = 'results'
# save_results(f"{PATH_RESULTS}/admm_results3.pkl", 
#              run_description=run_description,
#              param_exec=param_exec,
#              result_gurobi=result_gurobi,
#              result_CS=result_CS,)


#%% ======================= Plotting Results =======================

# # See ADMM solver performances
# visualize_admm_performance({admm_CS, result_CS},
#                            runtime_gurobi, cost_gurobi,
                        #    save_path="Figures/ADMM")

visualize_admm_performance({"ADMM CS": (admm_CS, result_CS), # update rho 10%
                            "ADMM CS CT": (admm_CS_CT, result_CS_CT)}, 
                            runtime_gurobi=runtime_gurobi,
                            cost_gurobi=cost_gurobi,
                            filename_suffix="rho_convergence")

#%% Show ADMM Details
visualize_admm_details({'ADMM CS': result_CS,
                        "ADMM CS CT": result_CS_CT},
                        combine_plots=True,
                        filename_suffix='_2')


#%% Show Power distribution
visualize_optimal_power_distribution(param_exec,
                                     Gurobi = power_distrib_gurobi,
                                     ADMM_Classical = power_distrib)


# %%
