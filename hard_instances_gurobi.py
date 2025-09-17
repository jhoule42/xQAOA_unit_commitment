# Check the new hard 0/1 Knapsack instances with Gurobi
# Julien-Pierre Houle
#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from classical_solver_UC import gurobi_knapsack_solver

from kp_utils import generate_inversely_strongly_correlated, generate_profit, generate_strong_spanner 
from kp_utils import generate_profit_spanner, generate_strongly_correlated


#%% ================== Loading Files ==================

# Get the list of 100 hard instances tested in the paper
list_instances_premlim_exp = np.loadtxt("preliminaryExperiment100Names.txt", dtype=str)

list_instances_n400 = [f.name for f in Path("problemInstances").iterdir() \
              if f.is_dir() and f.name.startswith("n_400")][:300]


def load_knapsack_instances(base_folder="problemInstances", list_instances=None):
    instances = {}

    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)

        if list_instances is not None and subfolder in list_instances:

            if not os.path.isdir(subfolder_path):  # Skip if not a folder
                continue

            file_path = os.path.join(subfolder_path, "test.in")
            if not os.path.exists(file_path):  # Skip if "test.in" doesn't exist
                continue

            with open(file_path, "r") as file:
                lines = file.readlines()
            
            n = int(lines[0].strip())  # Number of items
            items = [tuple(map(int, line.strip().split())) for line in lines[1:n+1]]  # List of (id, value, weight)
            capacity = int(lines[n+1].strip())  # Knapsack capacity

            # Extract values and weights separately
            values = [item[1] for item in items]
            weights = [item[2] for item in items]

            instances[subfolder] = {
                "num_items": n,
                "values": values,
                "weights": weights,
                "capacity": capacity
            }
    return instances


# %% =============== Solve OLD Hard KP instances with Gurobi =================

# Define problem parameters
capacity = 10000
nb_units = 400

# Generate profit and weight values
val_profit, weight_profit = generate_profit(nb_units)

# Generate different instances
instances = {
    "strongly_correlated": generate_strongly_correlated(nb_units),
    "strong_spanner": generate_strong_spanner(nb_units),
    "profit_spanner": generate_profit_spanner(nb_units),
    "inversely_strongly_correlated": generate_inversely_strongly_correlated(nb_units),
}

# Loop through each instance type and solve the knapsack problem
results = {}
for instance_name, (values, weights) in instances.items():
    print(f"Solving knapsack for {instance_name} instance...")

    result = gurobi_knapsack_solver(
        values=values,
        weights=weights,
        capacity=capacity,
        optimality_gap=1e-6,
        feasibility_tolerance=1e-4,
        integer_tolerance=1e-2,
        verbose=True  # Set to True for detailed logs
    )
    tracker = result['tracker']

    results[instance_name] = result  # Store results for later analysis
print("All instances solved!")



#%%

# Plot the execution time for each instance
instance_names = list(results.keys())
execution_times = [results[name]['runtime'] for name in instance_names]

plt.figure(figsize=(10, 8))

for result in results:
    tracker = results[result]['tracker']
    tracker.times.append(tracker.times[-1] + 1e-6)  # Add a small value to avoid log(0)
    tracker.gaps.append(0)
    plt.plot(tracker.times, tracker.gaps, label=result)

plt.axhline(y=1e-6, color='k', linestyle='--', label="Optimality Gap = 1e-6")
plt.xlabel("Time (s)")
plt.ylabel("Optimality Gap")
plt.title("Optimality Gap vs Runtime")
plt.xscale("log")  # Optional: Log scale for better visualization
plt.yscale("log")  # Optional: Log scale for better visualization
plt.grid()
plt.xlim(0, 2e2)
plt.legend()
plt.show()

# %% =============== Solve NEW Hard KP instances with Gurobi =================

def solve_all_instances():
    instances = load_knapsack_instances(base_folder="problemInstances",
                                        list_instances=list_instances_n400)
    solutions = {}

    for name, instance in instances.items():
        print(f"Solving instance: {name}...")

        try:
            result = gurobi_knapsack_solver(
                values=instance["values"],
                weights=instance["weights"],
                capacity=instance["capacity"],
                optimality_gap=1e-6,
                feasibility_tolerance=1e-4,
                integer_tolerance=1e-2,
                time_limit=120,
                verbose=True,  # Set to True for detailed logs
            )
            solutions[name] = result
            print(f"  Solution: {result}")
        except Exception as e:
            print(f"  Error solving {name}: {e}")

    return solutions

# Run the solver on all problem instances
solutions = solve_all_instances()
print("All instances solved!")

#%%
# Plot the execution time for each instance
plt.figure(figsize=(10, 8))

for result in solutions:
    tracker = solutions[result]['tracker']
    try:
        if tracker.times[-1] > 1e0:
            plt.plot(tracker.times, tracker.gaps)
    except:
        continue

plt.axhline(y=1e-6, color='k', linestyle='--', label="Optimality Gap = 1e-6")
plt.xlabel("Time (s)")
plt.ylabel("Optimality Gap (%)")
plt.title("Optimality Gap vs Runtime")
plt.xscale("log")  # Optional: Log scale for better visualization
plt.yscale("log")  # Optional: Log scale for better visualization
plt.xlim(right=2e2)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"figures/optimality_gap__hard_instances_new.png")
plt.close()


# Get the number of instances that took more than 110 seconds
nb_max_time = 0
for result in solutions:
    tracker = solutions[result]['tracker']
    try:
        if tracker.times[-1] > 110:
            nb_max_time += 1
    except:
        continue
print(nb_max_time)