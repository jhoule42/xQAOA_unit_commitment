# Author: Julien-Pierre
#%%%
import numpy as np
import matplotlib.pyplot as plt

# Data
S = np.arange(1, 100, 10)
N = np.floor(np.log2(S)) + 1
nb_stock = 150

# Plot styling
plt.style.use('default')
plt.figure(figsize=(10, 6))

# Plotting the two algorithms
plt.plot(S, N*nb_stock, label='Discretized QUBO', linestyle='--', linewidth=2, color='red')
plt.plot(S, S, label='Quantum ADMM', linestyle='-', linewidth=2, color='green')

# Highlighting the area where our novel algorithm is better
# plt.fill_between(S, S, N*nb_stock, where=(S < N*nb_stock), interpolate=True, color='green', alpha=0.3)

# Labels and Title
plt.title('Qubit Requirement Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Number of Units', fontsize=14)
plt.ylabel('Number of Qubits', fontsize=14)

# Customize the ticks and axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


# Legend
plt.legend(loc='upper left', fontsize=12)

# Show the plot
plt.show()

# %%
