""" Code to run the ADMM algorithm to evaluate the convergence depenging of rho parameters."""
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

from UC.scripts.utils.models import *
from UC.scripts.utils.utils import *
from UC.scripts.solvers.classical_solver_UC import *
from UC.scripts.solvers.admm_optimizer import ADMMParameters, ADMMOptimizer


#%% ============================ Parameters ============================

loads = [80, 150, 180, 200]  # Total power demand
n_units = [4, 6, 8, 12]
generate_logs = False

dict_admm_opt = {}
dict_admm_results = {}


for idx, n_units in enumerate(n_units):
    print(f"Number of units: {n_units}")

    L = loads[idx]
    A, B, C, p_min, p_max = generate_units(N=n_units)
    param_exec = {"L": L, "n_units":n_units, "A": A, "B": B,
                  "C": C, "p_min": p_min, "p_max": p_max}

    # Generate quadratic program for Unit Commitment
    qp_UC = create_uc_model(A, B, C, L, p_min, p_max)


    # TODO: Make this more condense
    print("\nGurobi Solver")
    bitstring_gurobi, power_distrib_gurobi, cost_gurobi, runtime_gurobi = classical_unit_commitment_qp(A, B, C, L, p_min, p_max)
    result_gurobi = [bitstring_gurobi, power_distrib_gurobi, cost_gurobi, runtime_gurobi]
    print(f"Bitstring: {bitstring_gurobi}")
    print(f"Power distribution: {power_distrib_gurobi}")
    print(f"Total Power Load: {np.sum(power_distrib_gurobi):.1f} | {L}")
    print(f"Cost: {cost_gurobi:.2f}")

    if generate_logs:
        logging.basicConfig(filename='logs/admm_update_rho_10%.log',
                            level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            filemode='w')
        
    # Set parameters
    admm_params = ADMMParameters(rho_initial=2500, factor_c=100, maxiter=100,
                                three_block=False, tol=1e-12, warm_start=True,
                                vary_rho=0) # update rho by 10%

    # CLASSICAL SOLVER
    admm_rho0 = ADMMOptimizer(params=admm_params, qubo_type='classical')
    result_rho0 = admm_rho0.solve(qp_UC)

    # Add to dictionary
    dict_admm_opt[f"rho0_{n_units}u"] = admm_rho0
    dict_admm_results[f"rho0_{n_units}u"] = result_rho0

    x_sol = result_rho0.x[:n_units] # extract binary string
    power_distrib = result_rho0.x[n_units:] # extract power distribution

    # Set small values (below threshold) to zero
    threshold = 1e-1
    power_distrib = np.where(np.abs(power_distrib) < threshold, 0, power_distrib)

    print("\nADMM Classical QUBO")
    print(f"Cost: {result_rho0.fval:.1f}")
    print(f"Bitstring: {x_sol}")
    print(f"Power distribution: {power_distrib}")
    print(f"Total Power Load: {np.sum(power_distrib):.1f} | {L}")

    # Close and remove logging handlers after the first block
    if generate_logs:
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.CRITICAL)  # Suppresses all logs except CRITICAL


    logging.basicConfig(filename='logs/admm_update_rho_residuals.log',
                        level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filemode='w')

    # Set parameters
    admm_params = ADMMParameters(rho_initial=2500, factor_c=100, maxiter=100,
                                 three_block=False, tol=1e-12, warm_start=True,
                                 vary_rho=1) # update rho by residuals

    # CLASSICAL SOLVER
    admm_rho1 = ADMMOptimizer(params=admm_params, qubo_type='classical')
    result_rho1 = admm_rho1.solve(qp_UC)

    # Add to dictionary
    dict_admm_opt[f"rho1_{n_units}u"] = admm_rho1
    dict_admm_results[f"rho1_{n_units}u"] = result_rho1

    x_sol = result_rho1.x[:n_units] # extract binary string
    power_distrib = result_rho1.x[n_units:] # extract power distribution

    # Set small values (below threshold) to zero
    threshold = 1e-1
    power_distrib = np.where(np.abs(power_distrib) < threshold, 0, power_distrib)


    print("\nADMM Classical QUBO")
    print(f"Cost: {result_rho1.fval:.1f}")
    print(f"Bitstring: {x_sol}")
    print(f"Power distribution: {power_distrib}")
    print(f"Total Power Load: {np.sum(power_distrib):.1f} | {L}")

    # Close and remove logging handlers after the first block
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.CRITICAL)  # Suppresses all logs except CRITICAL


#%% ======================= Saving Results =======================

def save_results(filename, **kwargs):
    """Save multiple variables to a pickle file."""
    with open(filename, "wb") as file:
        pickle.dump(kwargs, file)
        print(f"Results saved to {filename}.")

# Save all relevant data
run_description = """Run ADMM algorithm using a classical QUBO solver to compare the performance
of updating rho by 10% (i) vs updating rho with residuals (ii)."""

# TODO: Change to save list since we itterate on various units
PATH_RESULTS = 'results'
save_results(f"{PATH_RESULTS}/admm_rho_conv.pkl", 
            run_description=run_description,
            param_exec=param_exec,
            dict_admm_opt=dict_admm_opt,
            dict_admm_results=dict_admm_results)


#%% ======================= Plotting Results =======================

# See ADMM solver performances
visualize_admm_performance({r"$\rho_{10\%}$ - 4 units": (dict_admm_opt['rho0_4u'], dict_admm_results['rho0_4u']), # update rho 10%
                            r"$\rho_{resid}$ - 4 units": (dict_admm_opt['rho1_4u'], dict_admm_results['rho1_4u']), # update rho residuals
                            r"$\rho_{10\%}$ - 8 units": (dict_admm_opt['rho0_8u'], dict_admm_results['rho0_8u']),
                            r"$\rho_{resid}$ - 8 units": (dict_admm_opt['rho1_8u'], dict_admm_results['rho1_8u']),
                            r"$\rho_{10\%}$ - 12 units": (dict_admm_opt['rho0_12u'], dict_admm_results['rho0_12u']),
                            r"$\rho_{resid}$ - 12 units": (dict_admm_opt['rho1_12u'], dict_admm_results['rho1_12u'])}, 
                            runtime_gurobi=runtime_gurobi,
                            cost_gurobi=cost_gurobi,
                            save_path="Figures/ADMM/rho_conv",
                            filename_suffix="rho_convergence")

#%% Show ADMM Details
visualize_admm_details({'Update rho 10%': dict_admm_results['rho0_8u'],
                        'Update rho residuals': dict_admm_results['rho1_8u']},
                        save_path="Figures/ADMM/rho_conv",
                        combine_plots=True,
                        filename_suffix='rho_convergence')



# Make a graph about the cost vs iteration and see when solution start to be valid
# plot_admm_cost(result_quantum, rho_init, beta, factor_c, maxiter, three_block, tol)




# %%
