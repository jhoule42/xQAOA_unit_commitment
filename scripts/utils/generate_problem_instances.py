from pathlib import Path
import numpy as np

def load_instance(file_path: str):
    file_path = str(file_path)
    with open(file_path, "r") as f:
        lines = f.readlines()

    n = int(lines[0].strip())  # Number of items
    items = [tuple(map(int, line.strip().split())) for line in lines[1:n+1]]  # value, weight
    capacity = int(lines[n+1].strip())  # Knapsack capacity

    values = [v for v, w in items]
    weights = [w for v, w in items]

    return {
        "file": file_path,
        "num_items": n,
        "values": values,
        "weights": weights,
        "capacity": capacity
    }
def load_first_n_nap_instances(folder_path: str, n: int = 10):
    folder = Path(folder_path)
    # Get all .NAP files (case-sensitive); use case-insensitive match if needed
    instance_files = list(folder.glob("*.knp"))[:n]

    instances = []
    for file_path in instance_files:
        instance = load_instance(file_path)
        instances.append(instance)

    return instances

def load_first_n_nap_instances(folder_path: str, n: int = 10):
    folder = Path(folder_path)
    # Get all .NAP files (case-sensitive); use case-insensitive match if needed
    instance_files = list(folder.glob("*.knap"))[:n]

    instances = []
    for file_path in instance_files:
        instance = load_instance(file_path)
        instances.append(instance)

    return instances
def generate_profit(n,d,frac):
    weights = np.random.randint(1, 1001, n)
    values = d * np.ceil(weights / d).astype(int)
    capacity = frac * np.sum(weights)
    return values, weights,capacity

def generate_random_ratio(n, frac,ratio_range = 0.005):
    weights = np.random.randint(300, 1001, n)
    ratios = np.random.uniform(1- ratio_range, 1, n)
    # ratios = np.random.normal(1, ratio_range, n)
    values = weights * ratios
    capacity = frac * np.sum(weights)
    return  values,weights,capacity

def generate_inversely_strongly_correlated(n,frac):
    values = np.random.randint(1, 1001, n)
    weights = values + np.random.choice([98, 102], n)
    capacity = frac * np.sum(weights)
    return values, weights, capacity

def get_knapsack_instance(
     profit=False,
    inversely_strongly_correlated=False,
    random_ratio=False,
    use_hard_data=False,# relevant when use_hard_data=True
    real_dataset_name="Hard",# relevant when use_hard_data=True
    num_hard_instances=20,# relevant when use_hard_data=True
    instance_index= 13, # relevant when use_hard_data=True 13 for 100 items, 12 for 50 items
    n=100,
    d=3,
    frac=0.09
):
    """
    Load or generate a knapsack problem instance.

    Parameters:
    - use_real_data: If True, load real instance; otherwise, generate synthetic.
    - real_dataset_name: Name of the dataset to load real instances from.
    - instance_index: Index of the instance to select from loaded instances.
    - num_hard_instances: How many real instances to load.
    - n: Number of items (used only when generating synthetic instances).
    - d: Parameter for `generate_profit` (used when generating synthetic).
    - frac: Fraction of total weight used to set capacity.
    - use_random_ratio: If True, use `generate_random_ratio`; otherwise use `generate_profit`.

    Returns:
    - n: Number of items
    - weights: np.ndarray of item weights
    - values: np.ndarray of item values
    - capacity: Capacity of the knapsack
    - frac: Capacity as a fraction of total weight
    """

    if use_hard_data:
        print('hard data')
        instances = load_first_n_nap_instances(real_dataset_name, num_hard_instances)
        print(instances)
        instance = instances[instance_index]
        n = instance['num_items']
        print('number of item',n)
        weights = np.array(instance['weights'])
        values = np.array(instance['values'])
       # print('weights',weights)
        capacity = np.array(instance['capacity'])
        frac = capacity / np.sum(weights)
        print('frac',frac)
    elif inversely_strongly_correlated:
        print('inversely_strongly_correlated')
        values, weights,capacity = generate_inversely_strongly_correlated(n,frac)
        capacity=frac*np.sum(weights)
        print('frac',frac)
    elif profit:
        print('profit')
        print('frac',frac)
        values, weights,capacity = generate_profit(n, d, frac)
        weights = np.array(weights)
        values = np.array(values)
        capacity=frac*np.sum(weights)
    elif random_ratio:
        print('random ratio')
        print('frac',frac)
        values, weights,capacity = generate_random_ratio(n, frac)
        weights = np.array(weights)
        values = np.array(values)
        capacity=frac*np.sum(weights)
        #print('weights',weights) 
    else:
        raise ValueError("Must specify one of the instance generation.")  
    return n, weights, values, capacity, frac