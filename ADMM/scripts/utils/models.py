# Author: Julien-Pierre Houle

from docplex.mp.model import Model
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
import numpy as np

def create_uc_model(A, B, C, L, p_min, p_max, add_cross_terms=False, lambda1=1):
    """Create a Docplex model for the Unit Commitment problem.

    Args:
        A (list[float]): Fixed cost coefficients for units.
        B (list[float]): Linear cost coefficients for units.
        C (list[float]): Quadratic cost coefficients for units.
        L (float): Total power demand.
        p_min (list[float]): Minimum power output for each unit.
        p_max (list[float]): Maximum power output for each unit.

    Returns:
        QuadraticProgram: A Qiskit-compatible quadratic program.
    """
    n_units = len(A)
    mdl = Model("unit_commitment")

    # Decision Variables
    y = mdl.binary_var_list(n_units, name="y")
    p = [mdl.continuous_var(lb=0, ub=p_max[i], name=f"p_{i}")
        for i in range(n_units)]

    # Objective Function
    total_cost = mdl.sum(
        (A[i] * y[i]) + (B[i] * p[i]) + (C[i] * (p[i] ** 2))
        for i in range(n_units)
    )
    mdl.minimize(total_cost)

    # Constraints
    mdl.add_constraint(mdl.sum(p) == L, "power_balance")
    for i in range(n_units):
        mdl.add_constraint(p[i] >= p_min[i] * y[i], f"min_power_{i}")
        mdl.add_constraint(p[i] <= p_max[i] * y[i], f"max_power_{i}")

    mdl.add_constraint(mdl.sum(y) >= 1)

    qp = from_docplex_mp(mdl)

    return qp


def cross_terms_matrix(qp, lambda1, p_min, p_max, L, update_power=[]):

    # initial matrix (no cross-terms)
    quad_objective = qp.objective.quadratic.to_array()
    n_units = len(p_min)

    if len(update_power) == 0:
        p_i = [(p_max[i] + p_min[i]) / 2 for i in range(n_units)]
    else:
        update_power = np.array(update_power)
        # Update optimal power distribution for next iteration
        off_units = (update_power == 0)  # Boolean array where True means the unit is off
        midrange_values = (np.array(p_min) + np.array(p_max)) / 2

        # Set the values of the off units to their midrange values
        updated_optimal_power = update_power.copy()
        updated_optimal_power[off_units] = midrange_values[off_units]

        print("Updated power dist:", updated_optimal_power)
        p_i = updated_optimal_power
    
    # quadratic terms
    for i in range(n_units):
        for j in range(n_units):
            if i != j: # avoid self-product
                power_product = p_i[i] * p_i[j]
                quad_objective[n_units+i, n_units+j] += 2 * lambda1 * power_product

    # linear terms
    lin_objective = qp.objective.linear.to_array()
    for i in range(n_units):
        lin_objective[n_units+i] += -2 * lambda1 * L * p_i[i]

    qp.objective.quadratic = quad_objective
    qp.objective.linear = lin_objective

    return qp


def model_prob_6():
    # Create a model
    mdl = Model("Example6")

    # Define variables
    v = mdl.binary_var(name="v")
    w = mdl.binary_var(name="w")
    t = mdl.binary_var(name="t")
    u = mdl.continuous_var(name="u")

    # Objective function
    mdl.minimize(v + w + t + 5 * (u - 2) ** 2)

    # Constraints
    mdl.add_constraint(v + 2 * w + t + u <= 3, "constraint_1")
    mdl.add_constraint(v + w + t >= 1, "constraint_2")
    mdl.add_constraint(v + w == 1, "constraint_3")

    qp = from_docplex_mp(mdl)
    return qp



def uc_old_model(A, B, C, lambda1, L, p_min, p_max, optimized_power=None):
    """
    Create a Quadratic Unconstrained Binary Optimization (QUBO) problem for the
    unit commitment problem.

    Args:
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
    Returns:
        QuadraticProgram: The formulated QUBO problem.
    """
    n_units = len(A)

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
    # print(qp.prettyprint())

    return qp
