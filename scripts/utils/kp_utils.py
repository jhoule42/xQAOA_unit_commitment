""" Utility functions usefull for calculations."""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import seaborn as sns


def generate_strongly_correlated(n=10):
    weights = np.random.randint(1, 1001, n)
    values = weights + 1000
    return values, weights


def generate_inversely_strongly_correlated(n=10):
    values = np.random.randint(1, 1001, n)
    weights = values + np.random.choice([98, 102], n)
    return values, weights


def generate_profit(n=10, d=3):
    weights = np.random.randint(1, 1001, n)
    values = d * np.ceil(weights / d).astype(int)
    return values, weights


def generate_strong_spanner(n=10):
    span_size = 20
    span_values, span_weights = generate_strongly_correlated(span_size)
    span_weights = np.ceil(2 * span_weights / 3).astype(int)
    span_values = np.ceil(2 * span_values / 3).astype(int)
    
    values, weights = [], []
    for _ in range(n):
        idx = np.random.randint(0, span_size)
        s = np.random.choice([1, 2, 3])
        values.append(s * span_values[idx])
        weights.append(s * span_weights[idx])
    
    return np.array(values), np.array(weights)


def generate_profit_spanner(n=10):
    span_size = 20
    span_values, span_weights = generate_profit(span_size)
    span_weights = np.ceil(2 * span_weights / 3).astype(int)
    span_values = np.ceil(2 * span_values / 3).astype(int)
    
    values, weights = [], []
    for _ in range(n):
        idx = np.random.randint(0, span_size)
        s = np.random.choice([1, 2, 3])
        values.append(s * span_values[idx])
        weights.append(s * span_weights[idx])
    
    return np.array(values), np.array(weights)


def lazy_greedy_knapsack(v, w, c):
    """
    Very Greedy algorithm for the Knapsack Problem.
    Items are selected purely based on the highest efficiency ratio.
    """
    r = np.array(v) / np.array(w)  # Efficiency ratio

    # Sort items by efficiency ratio in descending order
    indices = np.argsort(-r)
    
    total_value = 0
    total_weight = 0
    selected_items = []

    for i in indices:
        if total_weight + w[i] <= c:
            total_weight += w[i]
            total_value += v[i]
            selected_items.append(i)

    # Create the bitstring representing the selected items
    bitstring = np.zeros(len(v), dtype=int)
    bitstring[selected_items] = 1

    return total_value, total_weight, ''.join(map(str, bitstring))


def lazy_greedy_knapsack(v, w, c):
    """
    Lazy Greedy algorithm for the Knapsack Problem.
    Implements Algorithm 1 from the paper exactly.
    """
    n = len(v)
    # Calculate efficiency ratios
    r = np.array(v) / np.array(w)
    
    # Sort indices by efficiency ratio (descending)
    # In case of ties, smaller index gets priority (stable sort)
    indices = np.argsort(-r, kind='stable')
    
    # Initialize variables as per the paper
    c_prime = c - w[indices[0]]  # Key difference: subtract first item's weight immediately
    j = 1  # Start from second item since we considered first item
    x = np.zeros(n, dtype=int)
    
    # Main loop following paper's algorithm
    while c_prime > 0 and j < n:
        x[indices[j]] = 1
        j += 1
        if j < n:
            c_prime = c_prime - w[indices[j]]
            
    # Calculate final value and weight
    value = sum(x[i] * v[i] for i in range(n))
    weight = sum(x[i] * w[i] for i in range(n))
    
    return value, weight, ''.join(map(str, x))

def very_greedy_knapsack(v, w, c):
    """
    Very Greedy algorithm for the Knapsack Problem.
    Implements Algorithm 2 from the paper exactly.
    """
    n = len(v)
    # Calculate efficiency ratios
    r = np.array(v) / np.array(w)
    
    # Sort indices by efficiency ratio (descending)
    # In case of ties, smaller index gets priority (stable sort)
    indices = np.argsort(-r, kind='stable')
    
    # Initialize variables as per the paper
    c_prime = c
    j = 0
    x = np.zeros(n, dtype=int)
    
    # Main loop following paper's algorithm
    while c_prime > 0 and j < n:
        # Inner while loop to skip items that don't fit
        while j < n and c_prime < w[indices[j]]:
            j += 1
            
        if j < n:  # If we found an item that fits
            x[indices[j]] = 1
            c_prime = c_prime - w[indices[j]]
            j += 1
    
    # Calculate final value and weight
    value = sum(x[i] * v[i] for i in range(n))
    weight = sum(x[i] * w[i] for i in range(n))
    
    return value, weight, ''.join(map(str, x))



def bruteforce_knapsack(values, weights, capacity, bit_mapping="regular", show_progress=True):
    """
    Brute-force solver for the knapsack problem.

    Parameters:
    values (list): List of item values.
    weights (list): List of item weights.
    capacity (int): Maximum weight capacity of the knapsack.
    bit_mapping (str): Either "regular" or "inverse" for bit interpretation.
    show_progress (bool): Whether to show the progress bar.

    Returns:
    list: Ranked solutions as a list of tuples (value, weight, bitstring).
    """
    import itertools
    from tqdm import tqdm

    n = len(values)
    ranked_solutions = []
    total_combinations = 2 ** n

    # Select iteration tool based on progress bar option
    iterator = (
        tqdm(itertools.product([0, 1], repeat=n), 
             total=total_combinations, 
             desc="Evaluating knapsack combinations")
        if show_progress else itertools.product([0, 1], repeat=n)
    )

    for subset in iterator:
        if bit_mapping == "regular":
            total_weight = sum(weights[i] * subset[i] for i in range(n))
            total_value = sum(values[i] * subset[i] for i in range(n))
        elif bit_mapping == "inverse":
            total_weight = sum(weights[i] * (1 - subset[i]) for i in range(n))
            total_value = sum(values[i] * (1 - subset[i]) for i in range(n))

        if total_weight <= capacity:
            ranked_solutions.append((total_value, total_weight, subset))
        else:
            ranked_solutions.append((0, 0, subset))

    ranked_solutions.sort(key=lambda x: x[0], reverse=True)
    ranked_solutions = [
        (int(value), int(weight), ''.join(map(str, bitstring))) 
        for value, weight, bitstring in ranked_solutions
    ]

    return ranked_solutions



def compute_min_D(B, C, L):
    """
    Compute the smallest D such that the capacity is non-negative.
    
    Parameters:
    B: list or np.array of B coefficients
    C: list or np.array of C coefficients
    L: threshold value for capacity
    
    Returns:
    optimal_D: The minimum D that satisfies the condition
    """
    B = np.array(B)
    C = np.array(C)
    
    # Compute the numerator and denominator
    numerator = 2 * L + np.sum(B / C)
    denominator = np.sum(1 / C)
    
    # Calculate D
    optimal_D = numerator / denominator
    return optimal_D



def from_Q_to_Ising(Q, offset):
    """Convert the matrix Q of Eq.3 into Eq.13 elements J and h"""
    n_qubits = len(Q)  # Get the number of qubits (variables) in the QUBO matrix
    # Create default dictionaries to store h and pairwise interactions J
    h = defaultdict(int)
    J = defaultdict(int)

    # Loop over each qubit (variable) in the QUBO matrix
    for i in range(n_qubits):
        # Update the magnetic field for qubit i based on its diagonal element in Q
        h[(i,)] -= Q[i, i] / 2
        # Update the offset based on the diagonal element in Q
        offset += Q[i, i] / 2
        # Loop over other qubits (variables) to calculate pairwise interactions
        for j in range(i + 1, n_qubits):
            # Update the pairwise interaction strength (J) between qubits i and j
            J[(i, j)] += Q[i, j] / 4
            # Update the magnetic fields for qubits i and j based on their interactions in Q
            h[(i,)] -= Q[i, j] / 4
            h[(j,)] -= Q[i, j] / 4
            # Update the offset based on the interaction strength between qubits i and j
            offset += Q[i, j] / 4
    # Return the magnetic fields, pairwise interactions, and the updated offset
    return h, J, offset



def energy_Ising(z, h, J, offset):
    """
    Calculate the energy of an Ising model given spin configurations.

    Parameters:
    - z: A dictionary representing the spin configurations for each qubit.
    - h: A dictionary representing the magnetic fields for each qubit.
    - J: A dictionary representing the pairwise interactions between qubits.
    - offset: An offset value.

    Returns:
    - energy: The total energy of the Ising model.
    """
    if isinstance(z, str):
        z = [(1 if int(i) == 0 else -1) for i in z]

    energy = offset # Initialize the energy with the offset term
    # Loop over the magnetic fields (h) for each qubit and update the energy
    for k, v in h.items():
        energy += v * z[k[0]]
    # Loop over the pairwise interactions (J) between qubits and update the energy
    for k, v in J.items():
        energy += v * z[k[0]] * z[k[1]]
    return energy


def sum_weight(bitstring, weights):
    weight = 0
    for n, i in enumerate(weights):
        if bitstring[n] == "1":
            weight += i
    return weight

def sum_values(bitstring, values):
    value = 0
    for n, i in enumerate(values):
        if bitstring[n] == "1":
            value += i
    return value



def reverse_bits(counts_dict):
    """
    Reverse the order of bits in the keys of the counts dictionary.
    
    Args:
        counts_dict (dict): A dictionary of bit string counts.
    
    Returns:
        dict: A new dictionary with the bits reversed.
    """
    new_counts = {}
    for bitstring, count in counts_dict.items():
        new_bitstring = bitstring[::-1]
        new_counts[new_bitstring] = count
    return new_counts





def get_hamming_distance(bitstring1, bitstring2):

    # Convert strings to numpy arrays of integers
    arr1 = np.array(list(bitstring1), dtype=int)
    arr2 = np.array(list(bitstring2), dtype=int)
    
    # Compute Hamming distance
    return np.sum(arr1 != arr2)



def solve_knapsack(weights, values, capacity, bitstrings):
    """
    Solve the 0/1 Knapsack problem by testing a list of bitstrings.
    
    Args:
    weights (list): List of item weights
    values (list): List of item values
    capacity (int): Maximum weight capacity of the knapsack
    bitstrings (list): List of bitstrings to test
    
    Returns:
    tuple: (best_value, best_bitstring, selected_items)
        - best_value: Maximum value achieved
        - best_bitstring: Bitstring that produced the maximum value
        - selected_items: Indices of items selected in the best solution
    """
    # Validate input
    if len(weights) != len(values):
        raise ValueError("Weights and values lists must be of equal length")
    
    # Initialize best solution tracking
    best_value = 0
    best_bitstring = None
    best_weight = None
    
    # Add tqdm progress bar for bitstrings
    for bitstring in tqdm(bitstrings, desc="Processing bitstrings", unit="bitstring"):
        # Validate bitstring length matches number of items
        if len(bitstring) != len(weights):
            raise ValueError(f"Bitstring {bitstring} length does not match number of items")
        
        # Calculate total weight and value for this bitstring
        current_weight = 0
        current_value = 0
        current_selected_items = []
        
        for i, bit in enumerate(bitstring):
            if bit == '1':
                current_weight += weights[i]
                current_value += values[i]
                current_selected_items.append(i)
        
        # Check if solution is valid and better than previous best
        if current_weight <= capacity and current_value > best_value:
            best_value = current_value
            best_bitstring = bitstring
            best_weight = current_weight
    
    return best_value, best_bitstring, best_weight


def extract_unique_bitstrings(big_dict):
    """
    Extract all unique bitstrings from a nested dictionary.
    
    Args:
    big_dict (dict): Nested dictionary with bitstrings in 'counts'
    
    Returns:
    list: List of unique bitstrings
    """
    # Use a set to automatically handle uniqueness
    unique_bitstrings = set()
    
    # Iterate through all sub-dictionaries
    for key, sub_dict in big_dict.items():
        # Check if 'counts' exists in the sub-dictionary
        if 'counts' in sub_dict:
            # Add all bitstrings from counts to the set
            unique_bitstrings.update(sub_dict['counts'].keys())
        else:
            try:
                unique_bitstrings.update(sub_dict['bitstrings'])
            except:
                continue
            
    # Convert set to list and return
    return list(unique_bitstrings)


# Helper function to convert NumPy types to standard Python types
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    else:
        return obj  # Return other objects as is




def get_value(bitstring, v, bit_mapping="regular"):
    """
    Compute the total value for a given Knapsack bitstring.

    Args:
        bitstring (str): A binary string representing the solution (e.g., "10101").
        v (list): List of values corresponding to the items.
        bit_mapping (str): Specifies the mapping mode ('regular' or 'inverse').

    Returns:
        float/int: Total value of the selected items.
    """
    if len(bitstring) != len(v):
        raise ValueError("Bitstring length must match the length of values.")
    
    if bit_mapping == "regular":
        return sum(int(bitstring[i]) * v[i] for i in range(len(bitstring)))
    elif bit_mapping == "inverse":
        return sum((1 - int(bitstring[i])) * v[i] for i in range(len(bitstring)))
    else:
        raise ValueError("Invalid bit_mapping mode. Use 'regular' or 'inverse'.")


def get_weight(bitstring, w, bit_mapping="regular"):
    """
    Compute the total weight for a given Knapsack bitstring.

    Args:
        bitstring (str): A binary string representing the solution (e.g., "10101").
        w (list): List of weights corresponding to the items.
        bit_mapping (str): Specifies the mapping mode ('regular' or 'inverse').

    Returns:
        float/int: Total weight of the selected items.
    """
    if len(bitstring) != len(w):
        raise ValueError("Bitstring length must match the length of weights.")
    
    if bit_mapping == "regular":
        return sum(int(bitstring[i]) * w[i] for i in range(len(bitstring)))
    elif bit_mapping == "inverse":
        return sum((1 - int(bitstring[i])) * w[i] for i in range(len(bitstring)))
    else:
        raise ValueError("Invalid bit_mapping mode. Use 'regular' or 'inverse'.")
    


def convert_bitstring_to_values(counts, v, w, c, filter_invalid_solutions=True):
    dict_bit_values = {}

    # Create a progress bar over the items in `counts`
    for bitstring, count in tqdm(counts.items(), desc="Processing bitstrings", total=len(counts)):
        
        if filter_invalid_solutions:
            if get_weight(bitstring, w) <= c:
                value = get_value(bitstring, v)
                dict_bit_values[value] = dict_bit_values.get(value, 0) + count
        else:
            value = get_value(bitstring, v)
            dict_bit_values[value] = dict_bit_values.get(value, 0) + count

    if not dict_bit_values:
        print("No valid solutions found!")

    return dict_bit_values


def compute_approximate_ratio(dict_bit_values, value_opt):
    # Compute the approximate ratio
    total = 0
    for values, counts in dict_bit_values.items():
        total += values * counts

    # sum all the counts in the histogram of valid solutions
    total_valid_shots = sum(dict_bit_values.values())
    unnorm_aprox_ratio = total / (total_valid_shots * value_opt)

    return unnorm_aprox_ratio, total_valid_shots

def probabilty_success(dict_bit_values, optimal_value):
    """ Compute the probability of sucess."""
    max_bitstring_val = max(dict_bit_values.keys())  # Find the highest key
    
    if max_bitstring_val == optimal_value:
        max_value_counts = dict_bit_values[max_bitstring_val]   # Get its associated value
        total_sum = sum(dict_bit_values.values())  # Sum all values
        return max_value_counts / total_sum

    # If there is no count for the optimal solution
    else:
        return 0