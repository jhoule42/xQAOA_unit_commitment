# **xQAOA Unit Commitment**
[![Python Versions](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Quantum](https://img.shields.io/badge/Quantum-QAOA-purple.svg)](https://qiskit.org/)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)](https://github.com/your-username/your-repo)


*Quantum Algorithm for Utility-Scale Power Grid Optimization*

<div align="center">
  <img src="image.png" alt="Unit Commitment Optimization" width="200">
</div>

---

This project implements **quantum and classical methods** to solve the **Unit Commitment (UC)** problem, which involves optimizing the scheduling of power generation units to meet electricity demand while minimizing costs. In this work we implement the **xQAOA** method, allowing to solve the UC problem on large scale quantum computers.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## **Method Description**

Two methods to solve the UC problem have been implemented as part of this project:

**1. ADMM (Alternating Direction Method of Multipliers)**
- Implementation inspired by the following work: http://arxiv.org/abs/2001.02069.
- The method provide good results when solving the UC problem, however relies to significantly on classical optimization and the quantum solver deal with trivial problems. It is then unlikely to provide any quantum advantage in our case.

**2. xQAOA**
- Implementation inspired by the following work: http://arxiv.org/abs/2108.08805.
- We have showned as part of this work that we can map the Knapsack problem the the UC problem using only one additional parameter to optimize.

## **File Structure**
```plaintext
xQAOA/
├── figures/                   # Generated figures and visualizations
├── runs/                      # Logs and outputs from various runs
├── scripts/
│   ├── experiments/           # Experimentation scripts
│   │   ├── qkp_evaluate_noise.py         # Evaluates noise impact on QKP
│   │   ├── qkp_optimal_parameters.py     # Finds optimal parameters for QAOA
│   │   ├── run_qkp_simulator_high_depth.py  # Runs QKP on a high-depth simulator (QAOA at p=2)
│   │   ├── solve_UC_knapsack.py          # Solves UC problem as a knapsack problem
│   │   └── visualize_LG_distribution.py  # Visualizes Lazy Greedy distribution
│   ├── solvers/               # Solver modules
│   │   └── qkp_solver.py      # Quantum Knapsack Problem solver with xQAOA
│   ├── utils/                 # Utility scripts
│   │   ├── kp_utils.py        # Knapsack-related utility functions
│   │   ├── visualize.py       # Visualization helpers
│   │   └── post-process_qkp_run.py  # Post-processes quantum run results
│   ├── run_qkp_fake_hardware.py  # Simulates hardware noise in QKP
│   ├── run_qkp_hardware.py       # Runs QKP on real quantum hardware
│   ├── run_qkp_simulator.py      # Runs QKP on quantum simulators
│   ├── solve_MUCP.py             # Solves the Multi-Unit Commitment Problem (not complete)
│   └── solve_MUCP_rolling_horizon.py  # Solves UC in a rolling horizon framework (not complete)
├── requirements.txt          # Python dependencies for the project
├── demo.ipynb                # Jupyter notebook demoing the main ideas of the xQAOA mixer (in progress)

## **License**

This project is licensed under the MIT License. See the LICENSE file for more details.
