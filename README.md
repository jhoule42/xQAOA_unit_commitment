# **xQAOA Unit Commitment**

*Quantum Algorithm for Utility-Scale Power Grid Optimization*

<div align="center">
  <img src="image.png" alt="Unit Commitment Optimization" width="200">
</div>

---

This project implements **quantum and classical methods** to solve the **Unit Commitment (UC)** problem, which involves optimizing the scheduling of power generation units to meet electricity demand while minimizing costs. It explores hybrid quantum-classical algorithms using **QAOA** and classical optimization techniques.

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

Three methods to solve the UC problem have been implemented as part of this project:

**1. Multi-variable Optimization**
- Inspired by the following papers: 
- Observed limitations: Results are far from ideal
- Requires optimizing as many additional parameters as the number of units
- Potentially challenging to scale

**2. ADMM (Alternating Direction Method of Multipliers)**
- Implementation inspired by the following work: http://arxiv.org/abs/2001.02069.
- The method provide good results when solving the UC problem, however relies to significantly on classical optimization and the quantum solver deal with trivial problems. It is then unlikely to provide any quantum advantage.

**3. xQAOA**
- Implementation inspired by the following work: http://arxiv.org/abs/2108.08805.
- Outperforms the two previous methods in accuracy. We have showned as part of this work that we can map the Knapsack problem the the UC problem using only one additional parameter to optimize. Can be easily scallable and we have implement the method on IBM_quebec quantum computer using up to 127 qubits.

## **File Structure**

*To be completed later...*

## **License**

This project is licensed under the MIT License. See the LICENSE file for more details.
