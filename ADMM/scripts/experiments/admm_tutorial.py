#%%
from docplex.mp.model import Model

from qiskit.primitives import Sampler
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import MinimumEigenOptimizer, ADMMOptimizer, ADMMParameters
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA

from UC.scripts.utils.utils import *

#%%
# Optimizer Setup
convex_optimizer = COBYLA()
qaoa = MinimumEigenOptimizer(QAOA(sampler=Sampler(), optimizer=convex_optimizer))
exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())


def build_toy_model():
    """ Build toy model for testing. """
    mdl = Model("unit_commitment_example")

    # Define variables
    v = mdl.binary_var(name="v")
    w = mdl.binary_var(name="w")
    t = mdl.binary_var(name="t")
    u = mdl.continuous_var(name="u")

    # Define objective and constraints
    mdl.minimize(v + w + t + 5 * (u - 2) ** 2)
    mdl.add_constraint(v + 2 * w + t + u <= 3, "cons1")
    mdl.add_constraint(v + w + t >= 1, "cons2")
    mdl.add_constraint(v + w == 1, "cons3")

    # Convert to a Qiskit Quadratic Program
    qp = from_docplex_mp(mdl)
    print(qp.prettyprint())
    return qp


#%% ====================== ADMM Solver ======================

def solve_with_admm(qp, params):
    admm = ADMMOptimizer(params=params)

    if admm.is_compatible(qp):
        result = admm.solve(qp)
        print(result.prettyprint())

    else:
        print("Problem is not compatible with ADMM.")

    return result


def solve_with_admm_qaoa(qp, qaoa, params):
    admm_q = ADMMOptimizer(params=params, qubo_optimizer=qaoa)
    result_q = admm_q.solve(qp)
    print(result_q.prettyprint())

    return result_q



#%% ============== Main Execution ==============

qp = build_toy_model()

# Define ADMM parameters
admm_params = ADMMParameters(rho_initial=1001, beta=1000, factor_c=900, 
                             maxiter=100, three_block=True, tol=1.0e-6)

# Solve using ADMM with classical solver
result_classical = solve_with_admm(qp, admm_params)
plot_residuals(result_classical.state.residuals)

# Solve using hybrid quantum-classical ADMM
result_quantum = solve_with_admm_qaoa(qp, qaoa, admm_params)
plot_residuals(result_classical.state.residuals)


# %%

#%% ================== RUN ADMM TOY PROBLEM ==================


# # Solve ADMM with classical exact solver
# result_classical = solve_with_admm(qp_toy, admm_params)
# plot_residuals(result_classical.state.residuals)


# # Solve ADMM with quantum exact solver
qaoa = MinimumEigenOptimizer(QAOA(sampler=Sampler(), optimizer=COBYLA()))

# result_quantum = solve_with_admm_qaoa(qp_toy, qaoa, admm_params)
# plot_residuals(result_classical.state.residuals)