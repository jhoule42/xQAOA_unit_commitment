""" Classical Solver used for to solve the Unit Commtiment problem.
Author: Julien-Pierre Houle """
#%%
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import minimize, BFGS


def gurobi_solver(A, B, C, L, p_min, p_max, verbose=False):
    """
    Solve the Unit Commitment problem using a quadratic programming solver (Gurobi).
    
    Args:
        A (list): Fixed cost coefficients for each unit.
        B (list): Linear operational cost coefficients for each unit.
        C (list): Quadratic operational cost coefficients for each unit.
        L (float): Total power demand (load).
        p_min (list): Minimum power output for each unit.
        p_max (list): Maximum power output for each unit.

    Returns:
        y_solution (list): Binary solution (on/off for each unit).
        p_solution (list): Power output solution for each unit.
        total_cost (float): Total cost of operation.
    """
    
    n_units = len(A)  # Number of power units

    # Create a new model
    model = gp.Model("unit_commitment")

    # Suppress output if verbose is False
    if not verbose:
        model.setParam('OutputFlag', 0)

    # Add binary decision variables for turning the units on/off
    y = model.addVars(n_units, vtype=GRB.BINARY, name="y")

    # Add continuous decision variables for the power output of each unit
    p = model.addVars(n_units, lb=0, name="p")

    # Objective function: minimize total cost
    total_cost = gp.quicksum( (A[i]*y[i]) + (B[i]*p[i]) + (C[i]*p[i]*p[i]) for i in range(n_units))
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Add constraint: Total power output must meet the load demand L
    model.addConstr(gp.quicksum(p[i] for i in range(n_units)) == L, name="power_balance")

    # Add constraints: Power output of each unit must respect on/off state and power bounds
    for i in range(n_units):
        model.addConstr(p[i] >= p_min[i] * y[i], name=f"min_power_{i}")
        model.addConstr(p[i] <= p_max[i] * y[i], name=f"max_power_{i}")

    # Increase precision in Gurobi
    model.setParam('MIPGap', 1e-12)  # Tighter gap for optimality
    model.setParam('FeasibilityTol', 1e-6)  # Tighter feasibility tolerance

    # Optimize the model
    model.optimize()


    if model.status == GRB.OPTIMAL:
        # Extract binary (y) and continuous (p) solutions
        y_solution = [y[i].x for i in range(n_units)]
        p_solution = [p[i].x for i in range(n_units)]
        total_cost = model.objVal

        y_solution = ''.join(['1' if num == 1.0 else '0' for num in y_solution]) # convert to string

        results = {}
        results['bitstring'] = y_solution
        results['power'] = p_solution
        results['cost'] = total_cost
        results['runtime'] = model.Runtime

        return results
    else:
        raise ValueError("Optimization failed")
    



def classical_power_distribution(x_sol, A, B, C, p_min, p_max, L, raise_error=True):
    """
    Distribute power among active units based on the binary solution from the QUBO problem.

    Args:
        x_sol (str): Binary solution string (from QAOA).
        B (list): Linear power coefficients for each unit.
        C (list): Quadratic power coefficients for each unit.
        p_min (list): Minimum power output for each unit.
        p_max (list): Maximum power output for each unit.
        L (float): Required total power load.

    Returns:
        tuple: Optimal power outputs and the associated cost.
    """

    nb_units = len(x_sol)
    active_units = [i for i, bit in enumerate(x_sol) if bit == '1']

    # Raise Value Errors if the optimisation is impossible
    if sum(p_max[i] for i in active_units) < L:
        if raise_error:
            raise ValueError("Total maximum power output of active units is less than required load L.")
        else:
            return [], 0
        
    if min(p_min) > L:
        if raise_error:
            raise ValueError("Minimum power output is more than the requiered load L.")
        else:
            return [], 0        

    if not active_units:
        if raise_error:
            raise ValueError("No active units, cannot distribute power.")
        return [], 0

    # Objective function: Minimize power generation cost for active units
    def objective(power):
        """Objective cost function to minimize."""
        cost = 0
        for i in active_units:
            cost += A[i] + (B[i]*power[i]) + (C[i]*(power[i]**2))
        return cost

    # Constraint to ensure total power output meets the load L
    def load_constraint(p):
        return np.sum(p) - L

    # Define bounds for power outputs
    bounds = [(p_min[i], p_max[i]) if x_sol[i] == '1' else (0, 0) for i in range(nb_units)]

    # Initial guess: even distribution of L among active units
    num_active_units = len(active_units)
    initial_guess = [0] * nb_units
    if num_active_units > 0:
        even_distribution = L / num_active_units
        for i in active_units:
            initial_guess[i] = min(max(even_distribution, p_min[i]), p_max[i])

    # Optimization with constraints
    constraints = [{'type': 'eq', 'fun': load_constraint}]
    
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError("Optimization failed:", result.message)

    # Check if the total power distribution matches the load L
    total_power = np.sum(result.x)
    if np.abs(total_power - L) > 1e-5:
        print(f"Warning: Total power distribution {total_power} does not match the load L={L}")

    return result.x, result.fun  # Return optimal power outputs and cost



def gurobi_knapsack_solver(
    values, 
    weights, 
    capacity, 
    verbose=False,
    time_limit=None,
    optimality_gap=1e-12,
    feasibility_tolerance=1e-9,
    integer_tolerance=1e-9,
    presolve=2,  # Aggressive presolve
    threads=None,  # Use all available threads by default
    method=None,  # Automatic method selection
    cuts=2,  # Aggressive cut generation
    heuristics=0.7,  # Increased heuristic effort
    node_method=None,  # Automatic node method
    scaling=0,  # Aggressive scaling
    branching=None,  # Automatic branching
    solution_limit=None  # No limit on solutions by default
):
    """
    Enhanced 0-1 Knapsack problem solver using Gurobi with extensive optimization parameters.
    
    Args:
    values (list): List of item values.
    weights (list): List of item weights.
    capacity (float): Maximum weight capacity of the knapsack.
    verbose (bool): Whether to show Gurobi optimization output.
    time_limit (float, optional): Maximum solving time in seconds.
    optimality_gap (float): MIP gap tolerance for optimality.
    feasibility_tolerance (float): Constraint feasibility tolerance.
    integer_tolerance (float): Integer feasibility tolerance.
    presolve (int): Presolve aggressiveness (0-2).
    threads (int, optional): Number of threads to use.
    method (int, optional): Simplex method to use.
    cuts (int): Cut generation aggressiveness (0-3).
    heuristics (float): Fraction of nodes where heuristics are used (0-1).
    node_method (int, optional): Node relaxation solution method.
    scaling (int): Scaling method (0-3).
    branching (int, optional): Branching method.
    solution_limit (int, optional): Limit on number of solutions to find.
    
    Returns:
    dict: Comprehensive optimization results
    """
    # Validate inputs
    n_items = len(values)
    if len(weights) != n_items:
        raise ValueError("Length of values and weights must be the same")
    
    # Create a new model
    model = gp.Model("knapsack")
    
    # Suppress output if verbose is False
    if not verbose:
        model.setParam('OutputFlag', 0)
    
    # Set computational parameters
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)
    
    # Precision and solving parameters
    model.setParam('MIPGap', optimality_gap)
    model.setParam('FeasibilityTol', feasibility_tolerance)
    model.setParam('IntFeasTol', integer_tolerance)
    
    # Presolve settings
    model.setParam('Presolve', presolve)
    
    # Threading and performance parameters
    if threads is not None:
        model.setParam('Threads', threads)
    
    # Method selection
    if method is not None:
        model.setParam('Method', method)
    
    # Cut generation
    model.setParam('Cuts', cuts)
    
    # Heuristics
    model.setParam('Heuristics', heuristics)
    
    # Node method
    if node_method is not None:
        model.setParam('NodeMethod', node_method)
    
    # Scaling
    model.setParam('ScaleFlag', scaling)
    
    # Branching
    if branching is not None:
        model.setParam('BranchDir', branching)
    
    # Solution limit
    if solution_limit is not None:
        model.setParam('SolutionLimit', solution_limit)
    
    # Add binary decision variables for item selection
    x = model.addVars(n_items, vtype=GRB.BINARY, name="x")
    
    # Objective function: maximize total value of selected items
    total_value = gp.quicksum(values[i] * x[i] for i in range(n_items))
    model.setObjective(total_value, GRB.MAXIMIZE)
    
    # Constraint: Total weight of selected items must not exceed capacity
    model.addConstr(
        gp.quicksum(weights[i] * x[i] for i in range(n_items)) <= capacity,
        name="weight_constraint"
    )
    
    # Optimize the model
    model.optimize()
    
    # Check optimization status
    if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        # Extract binary solution
        x_solution = [x[i].x for i in range(n_items)]
        
        # Convert to bitstring and get selected items
        bitstring = ''.join(['1' if num == 1.0 else '0' for num in x_solution])
        selected_items = [i for i in range(n_items) if x_solution[i] > 0.5]
        
        # Calculate total value and weight of selected items
        total_value = sum(values[i] for i in selected_items)
        total_weight = sum(weights[i] for i in selected_items)
        
        # Prepare comprehensive results
        results = {
            'bitstring': bitstring,
            'selected_items': selected_items,
            'total_value': total_value,
            'total_weight': total_weight,
            'runtime': model.Runtime,
            'status': model.status,
            'gap': model.MIPGap,
            'bound': model.ObjBound,
            'obj_val': model.ObjVal
        }
        return results
    else:
        raise ValueError(f"Optimization failed with status: {model.status}")

# # Example usage demonstrating parameter customization
# if __name__ == "__main__":
#     # Sample problem
#     values = [60, 100, 120]
#     weights = [10, 20, 30]
#     capacity = 50
    
#     # Different parameter configurations
#     results1 = gurobi_knapsack_solver(
#         values, weights, capacity, 
#         verbose=True,  # Show detailed output
#         time_limit=60,  # 60 seconds max solving time
#         optimality_gap=1e-6  # More relaxed optimality criterion
#     )
    
#     print("Optimization Results:")
#     for key, value in results1.items():
#         print(f"{key}: {value}")

#%%

# # Sample problem: 
# # Items with their values and weights
# values = [60, 100, 120]     # Values of items
# weights = [10, 20, 30]      # Weights of items
# capacity = 50               # Knapsack capacity

# # Solve the knapsack problem
# result = gurobi_knapsack_solver(values, weights, capacity, verbose=True)

# print("Optimization Results:")
# print(f"Bitstring: {result['bitstring']}")
# print(f"Selected Items: {result['selected_items']}")
# print(f"Total Value: {result['total_value']}")
# print(f"Total Weight: {result['total_weight']}")
# print(f"Runtime: {result['runtime']} seconds")



#%% ---------------------- PLOT GRAPH ----------------------

# # Classical Solving Time by number of Units
# nb_units_range = np.arange(10, 1000, 10)
# runtime_gurobi_arr = np.zeros(nb_units_range.size)

# for idx, nb_units in enumerate(nb_units_range):
#     A, B, C, p_min, p_max = generate_units(N=nb_units)

#     max_power = np.sum(p_max)
#     min_power = np.min(p_min)
#     L = np.random.uniform(min_power*1.2, max_power*0.9)

#     y_solution, p_solution, total_cost, runtime = classical_unit_commitment_qp(A, B, C, L, p_min, p_max)

#     runtime_gurobi_arr[idx] = runtime

# # %%

# # Plotting the runtime for Gurobi
# plt.figure(figsize=(10, 6))
# plt.plot(nb_units_range, runtime_gurobi_arr, label='Gurobi Solver', marker='o', linestyle='-', color='blue')

# # Adding title and labels
# plt.title('Gurobi Solver Runtime vs Number of Units - UC problem')
# plt.xlabel('Number of Units')
# plt.ylabel('Runtime (sec)')

# # Adding grid, legend, and aesthetics
# plt.grid(True)
# plt.legend(loc='upper left')
# plt.yscale('log')

# # Show the plot
# plt.savefig('Figures/gurobi_solver_runtime.png', bbox_inches='tight')
# plt.close()

# %%