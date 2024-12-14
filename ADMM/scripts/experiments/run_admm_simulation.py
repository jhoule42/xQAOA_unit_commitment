""" Code to run the ADMM algorithm."""
#%%
import pickle
import logging
import numpy as np

from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer, ADMMOptimizer, ADMMParameters
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeAuckland
from qiskit_ibm_runtime import SamplerV2

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

from UC.scripts.utils.utils import *
from UC.scripts.utils.visualize import *
from UC.scripts.utils.models import *
from UC.scripts.solvers.classical_solver_UC import *
from UC.scripts.solvers.admm_optimizer import ADMMParameters, ADMMOptimizer


#%% ============================ Unit Commitment Model ============================

LOG_PATH = '/Users/julien-pierrehoule/Documents/Stage/T3/Code/UC/logs'
PATH_RESULTS = 'results'

L = 100  # Total power demand
n_units = 4 # Number of units

A, B, C, p_min, p_max = generate_units(N=n_units)
param_exec = {"L": L, "n_units":n_units,
              "A": A, "B": B, "C": C,
              "p_min": p_min, "p_max": p_max}

# Generate quadratic program
qp_UC = create_uc_model(A, B, C, L, p_min, p_max)


#%% ======================= Classical Solver Gurobi =======================
print("\nGurobi Solver")
results_gurobi = gurobi_solver(A, B, C, L, p_min, p_max)
print(f"Bitstring: {results_gurobi['bitstring']}")
print(f"Power distribution: {results_gurobi['power']}")
print(f"Total Power Load: {np.sum(results_gurobi['power']):.1f} | {L}")
print(f"Cost: {results_gurobi['cost']:.2f}")


#%% ========================= ADMM CLASSICAL QUBO SOLVER =========================
logging.basicConfig(filename=f"{LOG_PATH}.results.log", level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set parameters
admm_params = ADMMParameters(rho_initial=5000, beta=1000, factor_c=100, maxiter=100,
                             three_block=False, tol=1e-12, warm_start=False, p=3, vary_rho=0)

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

#%%
# # Close and remove logging handlers after the first block
# for handler in logging.root.handlers[:]:
#     handler.close()
#     logging.root.removeHandler(handler)
# logging.basicConfig(level=logging.CRITICAL)  # Suppresses all logs except CRITICAL

#%% ========================= ADMM BASIC QUANTUM SOLVER =========================

param_history, value_history = [], []

def callback(mean, params, means, std):
    """Capture optimization progress."""
    param_history.append(params)
    value_history.append(mean)

def extract_params(param_history, p):
    """Extract gamma and beta values for each step."""
    gammas = [params[:p] for params in param_history]
    betas = [params[p:2 * p] for params in param_history]
    return gammas, betas

# Set parameters
admm_params = ADMMParameters(rho_initial=2500, beta=1000, factor_c=100, maxiter=100,
                             three_block=False, tol=1e-12, warm_start=True, p=3)


p = 2
# TODO: Will soon be depreciated, need to change it
qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, callback=callback)
qaoa_optimizer = MinimumEigenOptimizer(qaoa)

admm_BQ = ADMMOptimizer(params=admm_params, qubo_optimizer=qaoa_optimizer, qubo_type='qaoa_basic')
result_BQ = admm_BQ.solve(qp_UC)

x_sol = result_BQ.x[:n_units] # extract binary string
power_dist_BQ = result_BQ.x[n_units:] # extract power distribution

print("\nADMM BASIC Quantum SOLVER")
print(f"Cost: {result_BQ.fval:.1f}")
print(f"Bitstring: {x_sol}")
print(f"Power distribution: {power_dist_BQ}")
print(f"Total Power Load: {np.sum(power_dist_BQ):.1f} | {L}")

# Extract the parameters from qaoa basic
gammas_history, betas_history = extract_params(param_history, p)


#%% ========================= ADMM ADVANCE QUANTUM SOLVER =========================

# Set parameters
admm_params = ADMMParameters(rho_initial=5000, beta=1000, factor_c=100, maxiter=100,
                             three_block=False, tol=1e-12, warm_start=True, p=5)

admm_AQ = ADMMOptimizer(params=admm_params, qubo_type='qaoa_advance')
result_AQ = admm_AQ.solve(qp_UC) # rub the optimizer
print(result_AQ.prettyprint())

x_sol = result_AQ.x[:n_units] # extract binary string
power_dist_AQ = result_AQ.x[n_units:] # extract power distribution
power_dist_AQ = np.where(np.abs(power_dist_AQ) < 1e-3, 0, power_dist_AQ)

print("\nADMM Advance Quantum")
print(f"Cost: {result_AQ.fval:.1f}")
print(f"Bitstring: {x_sol}")
print(f"Power distribution: {power_dist_AQ}")
print(f"Total Power Load: {np.sum(power_dist_AQ):.1f} | {L}")

result_scipy = result_AQ.state.results_qaoa_optim


#%% ========================= ADMM ADVANCE QUANTUM SOLVER HARDWARE =========================

# Set parameters
p = 3
admm_params = ADMMParameters(rho_initial=2500, beta=1000, factor_c=100, maxiter=100,
                             three_block=False, tol=1e-12, warm_start=True, p=3,
                             gammas0=np.linspace(0, 2*np.pi, p),
                             betas0=np.linspace(np.pi, 0, p))

# backend = FakeManilaV2()
backend = FakeAuckland()
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
sampler = SamplerV2(backend)

admm_AQ_F = ADMMOptimizer(params=admm_params,
                        qubo_type='qaoa_advance',
                        hardware_execution=True,
                        backend=backend,
                        pass_manager=pm,
                        sampler=sampler,)

result_AQ_F = admm_AQ_F.solve(qp_UC)
print(result_AQ_F.prettyprint())

#%%






x_sol = result_AQ_F.x[:n_units] # extract binary string
power_dist_AQ_F = result_AQ_F.x[n_units:] # extract power distribution
power_dist_AQ_F = np.where(np.abs(power_dist_AQ_F) < 1e-3, 0, power_dist_AQ_F)

print("\nADMM Advance Quantum")
print(f"Cost: {result_AQ_F.fval:.1f}")
print(f"Bitstring: {x_sol}")
print(f"Power distribution: {power_dist_AQ_F}")
print(f"Total Power Load: {np.sum(power_dist_AQ_F):.1f} | {L}")

result_scipy = result_AQ_F.state.results_qaoa_optim


#%% ======================= Saving Results =======================

def save_results(filename, **kwargs):
    """Save multiple variables to a pickle file."""
    with open(filename, "wb") as file:
        pickle.dump(kwargs, file)
        print(f"Results saved to {filename}.")

# Save all relevant data
run_description = """Run ADMM Simulation #1"""

save_results(f"{PATH_RESULTS}/admm_results3.pkl", 
             run_description=run_description,
             param_exec=param_exec,
             result_gurobi=results_gurobi,
             result_CS=result_CS,
             result_BQ=result_BQ,
             result_AQ=result_AQ, 
             value_history=value_history, 
             param_history=param_history, 
             gammas_history=gammas_history, 
             betas_history=betas_history, 
             power_dist_BQ=power_dist_BQ,
             )


#%% ======================= Plotting Results =======================

# See ADMM solver performances
visualize_admm_performance(results_gurobi['runtime'], results_gurobi['cost'],
                           admm_CS, result_CS, 
                           admm_BQ, result_BQ,
                           admm_AQ, result_AQ,
                           admm_AQ, result_AQ,
                           save_path="Figures/ADMM")

#%% Show ADMM Details
# visualize_admm_details({'Advance Quantum': result_AQ,
#                         'Basic Quantum': result_BQ,
#                         'Classical Qubo': result_CS},
#                         save_path="Figures/ADMM",
#                         combine_plots=True,
#                         filename_suffix='_2')

visualize_admm_details({
                        'Classical Qubo': result_CS},
                        save_path="Figures/ADMM",
                        combine_plots=True,
                        filename_suffix='_2')


#%% Show Power distribution
visualize_optimal_power_distribution(param_exec,
                                     Gurobi = results_gurobi["power"],
                                    #  Advance_Quantum = power_dist_AQ,
                                     ADMM_Classical = power_distrib,
                                     Basic_Quantum = power_dist_BQ
                                     )


#%% Ranking the bitstring
# ATTENTION: l'algo fait d'autre calcul après le dernier QAOA
# Aussi il minimise x* --> c'est pas clair que c'est équivalent à l'autre problème!
# Valide uniquement pour advance quantum
counts = admm_AQ._state.counts
count_perf, count_rank = evaluate_perf_algo(counts, A, B, C, p_min, p_max, L)
plot_custom_histogram(counts,
                      highlighted_outcome=results_gurobi['bitstring'],
                      max_bitstrings=32,
                      bitstring_rankings=count_rank)

#%%


# TODO: Investigate the impact of changing parameters range
p = admm_BQ._params.p
init_gamma = np.linspace(0, 2*np.pi, p)
init_beta = np.linspace(np.pi, 0, p)
plot_qaoa_parameter_evolution(gammas_history, betas_history, p, init_gamma, init_beta)

plot_value_history(value_history)

# %%
