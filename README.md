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
- Inspired by the following papers: XXX
- Observed limitations: Results are far from ideal
- Requires optimizing as many additional parameters as the number of units
- Potentially challenging to scale

**2. ADMM (Alternating Direction Method of Multipliers)**
- Provides good results, but with significant limitations
- Relies too strongly on the classical solver
- Quantum solver solves trivial problems
- Unlikely to demonstrate quantum advantage

**3. xQAOA**
- Originally inspired by the following paper: XXX
- Outperforms the two previous methods
- Can be adapted to optimize with fewer additional parameters
- Does not require optimization for each simulated unit

## **File Structure**

*To be completed later...*

## **License**

This project is licensed under the MIT License. See the LICENSE file for more details.
