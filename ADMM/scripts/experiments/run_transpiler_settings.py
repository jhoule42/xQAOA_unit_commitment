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

import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

from UC.scripts.utils.models import *
from UC.scripts.utils.utils import *
from UC.scripts.solvers.classical_solver_UC import *
from UC.scripts.solvers.admm_optimizer import ADMMParameters, ADMMOptimizer

#%% ============================ Parameters ============================

# TODO: Make generate_units create loads according to n_units
loads = [100]  # Total power demand
nb_units = [4]
generate_logs = False
transpiler_optimization_level = [0, 1, 2, 3]

# initialize results dictionary
dict_admm_opt = {}
dict_admm_results = {}

for idx, opt_level in enumerate(transpiler_optimization_level):
    print(f"Optimization Level: {opt_level}")

    for n_units in nb_units:
        print(f"Number of units: {n_units}")

        L = loads[0]
        A, B, C, p_min, p_max = generate_units(N=n_units)
        param_exec = {"L": L, "n_units":n_units, "A": A, "B": B,
                      "C": C, "p_min": p_min, "p_max": p_max}

        # Generate quadratic program for Unit Commitment
        qp_UC = create_uc_model(A, B, C, L, p_min, p_max)


        # TODO: Make this more dense
        print("\nGurobi Solver")
        bitstring_gurobi, power_distrib_gurobi, cost_gurobi, runtime_gurobi = classical_unit_commitment_qp(A, B, C, L, p_min, p_max)
        result_gurobi = [bitstring_gurobi, power_distrib_gurobi, cost_gurobi, runtime_gurobi]
        print(f"Bitstring: {bitstring_gurobi}")
        print(f"Power distribution: {power_distrib_gurobi}")
        print(f"Total Power Load: {np.sum(power_distrib_gurobi):.1f} | {L}")
        print(f"Cost: {cost_gurobi:.2f}")

        if generate_logs:
            logging.basicConfig(filename='logs/admm_transpiler_settings.log',
                                level=logging.DEBUG,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                filemode='w')

        # Set parameters
        admm_params = ADMMParameters(rho_initial=2500, beta=1000, factor_c=100, maxiter=100,
                                     three_block=False, tol=1e-12, warm_start=True, p=3)
        
        # backend = FakeManilaV2()
        backend = FakeAuckland()
        pm = generate_preset_pass_manager(backend=backend, optimization_level=opt_level)
        sampler = SamplerV2(backend)

        admm_AQ_F = ADMMOptimizer(params=admm_params,
                                qubo_type='qaoa_advance',
                                hardware_execution=True,
                                backend=backend,
                                pass_manager=pm,
                                sampler=sampler,)

        result_AQ_F = admm_AQ_F.solve(qp_UC)
        print(result_AQ_F.prettyprint())

        # Save data to dictionary
        dict_admm_opt[f'transpile_opt{opt_level}'] = admm_AQ_F
        dict_admm_results[f'transpile_opt{opt_level}'] = result_AQ_F

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
run_description = """Run ADMM algorithm using a FakeBackend to compare the performance
of the algorithm when changing the transpiler settings."""

# TODO: Change to save list since we itterate on various units
PATH_RESULTS = '/Users/julien-pierrehoule/Documents/Stage/T3/Code/UC/results'
save_results(f"{PATH_RESULTS}/admm_transpile_settings.pkl", 
            run_description=run_description,
            param_exec=param_exec, # !!! make this a dict !!!
            dict_admm_opt=dict_admm_opt,
            dict_admm_results=dict_admm_results)


#%% ======================= Plotting Results =======================

# See ADMM solver performance
visualize_admm_performance({"Optimisation: 0 - 4 units": (dict_admm_opt['transpile_opt0'], dict_admm_results['transpile_opt0']),
                            "Optimisation: 1 - 4 units": (dict_admm_opt['transpile_opt1'], dict_admm_results['transpile_opt1']),
                            "Optimisation: 2 - 4 units": (dict_admm_opt['transpile_opt2'], dict_admm_results['transpile_opt2']),
                            "Optimisation: 3- 4 units": (dict_admm_opt['transpile_opt3'], dict_admm_results['transpile_opt3'])}, 
                            runtime_gurobi=runtime_gurobi,
                            cost_gurobi=cost_gurobi,
                            save_path="/Users/julien-pierrehoule/Documents/Stage/T3/Code/UC/figures",
                            filename_suffix="_transpiler_settings")


#%% Show ADMM Details
visualize_admm_details({"Optimisation: 0 - 4 units": (dict_admm_opt['transpile_opt0'], dict_admm_results['transpile_opt0']),
                        "Optimisation: 1 - 4 units": (dict_admm_opt['transpile_opt1'], dict_admm_results['transpile_opt1']),
                        "Optimisation: 2 - 4 units": (dict_admm_opt['transpile_opt2'], dict_admm_results['transpile_opt2']),
                        "Optimisation: 3- 4 units": (dict_admm_opt['transpile_opt3'], dict_admm_results['transpile_opt3'])},
                            save_path="/Users/julien-pierrehoule/Documents/Stage/T3/Code/UC/figures",
                        combine_plots=True,
                        filename_suffix='transpiler_settings')



# %%
