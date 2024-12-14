""" Run the QAOA multi-variables hybrid algorithm.
Author: Julien-Pierre Houle """
#%%
import numpy as np
from docplex.mp.model import Model
from scipy.optimize import minimize, BFGS

from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization import QuadraticProgram

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from UC.scripts.utils.utils import *
from UC.scripts.solvers.classical_solver_UC import *

#%% ============================ CREATE QUBO ============================


def unit_commitment_qubo(n_units, A, B, C, lambda1, L, p_min, p_max, optimized_power=None):
    """
    Create a Quadratic Unconstrained Binary Optimization (QUBO) problem for the
    unit commitment problem.

    Args:
        n_units (int): Number of power units.
        A (list): Linear cost coefficients for each unit.
        B (list): Linear power coefficients for each unit.
        C (list): Quadratic power coefficients for each unit.
        lambda1 (float): Penalty weight for power imbalance.
        lambda2 (float): Penalty weight for minimum power constraints.
        lambda3 (float): Penalty weight for maximum power constraints.
        L (float): Required total power load.
        p_min (list): Minimum power output for each unit.
        p_max (list): Maximum power output for each unit.
        optimized_power (list, optional): Optimized power values.
        s1 (list, optional): Additional slack for minimum power constraints.
        s2 (list, optional): Additional slack for maximum power constraints.

    Returns:
        QuadraticProgram: The formulated QUBO problem.
    """

    # Create a quadratic program
    qp = QuadraticProgram()

    # Define binary variables (y_i for each unit)
    [qp.binary_var(name=f'y_{i}') for i in range(n_units)]

    # Use optimized power if provided, otherwise default to average power between p_min and p_max
    if optimized_power is None:
        p_i = [(p_max[i] + p_min[i]) / 2 for i in range(n_units)]
    else:
        p_i = optimized_power

    # Initialize linear and quadratic terms
    linear_terms = {f'y_{i}': A[i] for i in range(n_units)}
    quadratic_terms = {}

    # Add power output terms
    for i in range(n_units):
        power_linear_term = B[i] * p_i[i]
        power_quadratic_term = C[i] * (p_i[i] ** 2)
        
        # Add linear term for power
        linear_terms[f'y_{i}'] += power_linear_term
        
        # Add quadratic term for power
        quadratic_terms[(f'y_{i}', f'y_{i}')] = quadratic_terms.get((f'y_{i}', f'y_{i}'), 0) + power_quadratic_term

    # Penalty terms for power imbalance
    for i in range(n_units):
        for j in range(n_units):
            if i != j:  # Avoid self-product
                power_product = p_i[i] * p_i[j]
                quadratic_terms[(f'y_{i}', f'y_{j}')] = quadratic_terms.get((f'y_{i}', f'y_{j}'), 0) + 2 * lambda1 * power_product

    # Add linear penalty terms for imbalance
    for i in range(n_units):
        linear_terms[f'y_{i}'] += -2 * lambda1 * L * p_i[i]

    # Add constant term (L^2) to the objective function
    qp.minimize(constant=lambda1 * (L**2), linear=linear_terms, quadratic=quadratic_terms)

    # Print the final QUBO formulation
    print(qp.prettyprint())

    return qp


def UC_model(A, B, C, L, p_min, p_max):
    n_units = len(A)  # Number of power units
    mdl = Model("unit_commitment")
    
    # Decision variables
    y = mdl.binary_var_list(n_units, name="y")
    p = [mdl.continuous_var(lb=0, name=f"p_{i}") for i in range(n_units)]
    
    # Objective function
    total_cost = mdl.sum((A[i]*y[i]) + (B[i]*p[i]) + (C[i]*(p[i] ** 2)) for i in range(n_units))
    mdl.minimize(total_cost)

    # Constraints
    mdl.add_constraint(mdl.sum(p) == L, "power_balance")
    epsilon = 1e-3  # Small tolerance
    for i in range(n_units):
        mdl.add_constraint(p[i] >= p_min[i] * y[i] - epsilon, f"min_power_{i}")
        mdl.add_constraint(p[i] <= p_max[i] * y[i] + epsilon, f"max_power_{i}")

    # 3. At least one unit must be on (if required)
    mdl.add_constraint(mdl.sum(y) >= 1, "Min_One_Active")

    qp = from_docplex_mp(mdl)
    print(qp.prettyprint())
    return qp



# Define the cost function
def cost_func_qaoa(params, ansatz, hamiltonian, estimator):
    """Evaluate the cost function using the estimator."""
    pub = (ansatz, hamiltonian, params)
    cost = estimator.run([pub]).result()[0].data.evs

    return cost


#%% ======================= MAIN HYBRID LOOP =======================

def run_hybrid_loop(iterations, L, p, lambda1, A, B, C, p_min, p_max, opt_power_classical,
                    plot_fig=True, display_count_ranking=True):
    """
    Runs a hybrid quantum-classical loop using QAOA for the unit commitment problem.

    Args:
        iterations (int): Number of hybrid loop iterations.
        L (float): Total power demand.
        p (int or list): QAOA depth, either a scalar or list of depths.
        lambda1 (float): Penalty term coefficient.
        A, B, C (array-like): Parameters for the unit commitment problem.
        p_min, p_max (array-like): Minimum and maximum power output per unit.
        plot_fig (bool): Whether to plot figures.
    """

    qaoa_cost = []
    p_min = np.array(p_min)
    p_max = np.array(p_max)

    # iterations = 1
    
    # If p is a scalar, convert it to a list for uniform iteration handling
    if isinstance(p, int):
        p = [p]

    # Iterate over the list of p values
    for depth in p:
        print(f"Running QAOA with depth p = {depth}")
        
        for it in range(iterations):
            print(f"------------ Iteration #{it+1} ------------")

            # Use updated optimal power from previous iteration
            if it == 0:
                updated_optimal_power = None

            # Quantum Optimization: Define the QUBO and Ising Hamiltonian
            qp = unit_commitment_qubo(n_units, A, B, C, lambda1, L, p_min, p_max, updated_optimal_power)
            cost_hamiltonian, offset = qp.to_ising()

            # Create QAOA ansatz and optimize
            circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=depth)
            pass_manager = generate_preset_pass_manager(3, AerSimulator())
            isa_circuit = pass_manager.run(circuit)

            # Initial parameter setup
            if it == 0:
                x0_params = np.empty(2 * depth, dtype=float)
                x0_params[0::2] = np.linspace(0, 2*np.pi, depth)  # Fill even indices with gammas
                x0_params[1::2] = np.linspace(np.pi, 0, depth)  # Fill odd indices with betas

            # Minimize the cost function
            estimator = Estimator()
            result_opt = minimize(cost_func_qaoa,
                                  x0=x0_params,
                                  args=(isa_circuit, cost_hamiltonian, estimator),
                                  method="COBYLA",
                                  options={'maxiter': 10000, 'disp':True})

            # Plot the optimization of the parameters
            if plot_fig:
                plot_optimization_results(depth, result_opt, x0_params)

            # Sampling the binary solution from the quantum circuit
            measured_circuit = isa_circuit.copy()
            measured_circuit.measure_all()
            exact_sampler = Sampler()
            isa_circuit = pass_manager.run(measured_circuit)
            pub = (isa_circuit, result_opt.x, 100000)
            job = exact_sampler.run([pub])
            result = job.result()
            pub_result = result[0]
            counts = pub_result.data.meas.get_counts()

            # Get the binary solution with the highest count
            x_sol = max(counts, key=counts.get)
            print(f"Binary solution from QAOA: {x_sol}")

            if plot_fig:
                if display_count_ranking:
                    count_perf, count_rank = evaluate_perf_algo(counts, A, B, C, p_min, p_max, L)

                plot_custom_histogram(counts, highlighted_outcome=x_solution, max_bitstrings=32,
                                      bitstring_rankings=count_rank)

            # Optimize the power distribution using the binary solution
            try:
                opt_power_quantum, cost = classical_power_distribution(x_sol, A, B, C, p_min, p_max, L)
                qaoa_cost.append(cost)  # Append the QAOA cost for the current iteration

                # Update optimal power distribution for next iteration
                off_units = (opt_power_quantum == 0)  # Boolean array where True means the unit is off
                midrange_values = (p_min + p_max) / 2

                # Set the values of the off units to their midrange values
                updated_optimal_power = opt_power_quantum.copy()
                updated_optimal_power[off_units] = midrange_values[off_units]
                print(f"Optimized power: {opt_power_quantum}")
                print(f"Updated Optimal Power (next iteration): {updated_optimal_power}")


                if plot_fig:
                    plot_optimal_power_distribution(p_min, p_max, opt_power_quantum, opt_power_classical,
                                                    A=A, B=B, C=C)
            except:
                print('Error trying to do the optimisation...\n\n')
                qaoa_cost.append(0)

    return qaoa_cost, result_opt


#%% ================== Parameters ==================

iterations = 1
L = 100  # Total power demand
p = 5  # Number of QAOA repetitions (layers)
n_units = 4
lambda1 = 1e4
PATH_FIGS = "UC/Figures/Multi-vars"


A, B, C, p_min, p_max = generate_units(N=n_units)

# Generate quadratic program
qp = UC_model(A, B, C, L, p_min, p_max)

# Classical Solver -- Gurobi
print('Classical Gurobi Solver')
x_solution, opt_power_classical, cost_classical, runtime = classical_unit_commitment_qp(A, B, C, L, p_min, p_max)



#%% ======================= IMPACT OF P VALUES (DEPTH) =======================

p_range = np.arange(1, 40, 4)
p_range = [5]

qaoa_cost, result_opt = run_hybrid_loop(iterations, L, p_range, lambda1, A, B, C, p_min, p_max, opt_power_classical,
                            plot_fig=True)


plot_cost_comparison_param(param_range=p_range,
                           qaoa_cost=qaoa_cost,
                           cost_classical=cost_classical,
                           param_name='p (Depth)',
                           filename=f'{PATH_FIGS}/cost_U{n_units}_p[{p_range[0]:.0f}-{p_range[-1]:.0f}]_lambda{lambda1:.0e}')



#%% ======================= IMPACT OF LAMBDA LOAD CONSTRAINT =======================

lambda_range = np.linspace(10, 1e10, 20)
p = 5
L = 100

list_cost_lambda = []

for lambda1 in lambda_range:
    qaoa_cost = run_hybrid_loop(iterations, L, p, lambda1, A, B, C, p_min, p_max, opt_power_classical,
                                plot_fig=True)
    list_cost_lambda.append(qaoa_cost)

list_cost = [item for sublist in list_cost_lambda for item in sublist]

plot_cost_comparison_param(param_range=lambda_range, qaoa_cost=list_cost,
                           cost_classical=cost_classical,
                           param_name='lambda',
                           filename=f'{PATH_FIGS}/cost_U{n_units}_p{p}_lambda-[{lambda_range[0]:.0e}-{lambda_range[-1]:.0e}]')
# %%
