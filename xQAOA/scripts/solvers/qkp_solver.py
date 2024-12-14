""" xQAOA Quantum Solver for Knapsack. """

import time
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from itertools import product


class QKPOptimizer:
    def __init__(self, v, w, c, mixer, p=1, run_hardware=False, backend=None,
                 sampler=None, pass_manager=None, optimal_solution=None,
                 generate_jobs=False, speedup_computation=True,
                 h_term=None, J_term=None):
        self.v = v
        self.w = w
        self.c = c
        self.mixer = mixer
        self.backend = backend
        self.run_hardware = run_hardware
        self.pass_manager = pass_manager
        self.optimal_solution = optimal_solution
        self.generate_jobs = generate_jobs
        self.speedup_computations = speedup_computation
        self.p = p
        self.h = h_term
        self.J = J_term
        

        self.n = len(v)
        self.best_bitstring = None
        self.best_value = -np.inf
        self.best_weight = -np.inf
        if sampler == None:
            self.sampler = StatevectorSampler()
        else:
            self.sampler = sampler
        if generate_jobs:
            self.list_transpile_qc = []
        self.dict_all_parameters = {}


    def logistic_bias(self, k):
        """Creates a biased initial distribution using the logistic function."""
        r = np.array(self.v) / np.array(self.w)
        C = (sum(self.w) / self.c) - 1
        return 1 / (1 + C * np.exp(-k * (r - r.mean())))
    

    def apply_cost_unitary(self, qc, gamma):
        """Applies the cost unitary UC(γ) to the quantum circuit."""
        for i, value in enumerate(self.v):
            qc.rz(-2 * gamma * value, i)


    def apply_cost_hamiltonian(self, qc, gammas):
        """ Cost Hamiltonian. """
        print('Cost Hamiltonian')
        h = self.h
        J = self.J

        # Normalize the Hamiltonian
        wmax = max(np.max(np.abs(list(h.values()))), np.max(np.abs(list(h.values()))))

        # Single-qubit terms
        for ki, v in h.items():
            qc.rz(2 * gammas * v / wmax, qubit=ki[0])

        # Two-qubit terms
        for kij, vij in J.items():
            qc.cx(kij[0], kij[1])
            qc.rz(2 * gammas * vij / wmax, kij[1])
            qc.cx(kij[0], kij[1])


    def apply_hourglass_mixer(self, qc, beta, p):
        """Applies the Hourglass mixer UBZX(β)."""
        for i, pi in enumerate(p):
            angle = 2 * np.arcsin(np.sqrt(pi))
            qc.ry(2 * angle, i)
            qc.rz(-2 * beta, i)
            qc.ry(-2 * angle, i)


    def apply_X_mixer(self, qc, beta):
        """Applies the standard QAOA X mixer."""
        for i in range(self.n):
            qc.rx(2 * beta, i)


    def apply_copula_mixer(self, qc, beta, p1, p2, theta):
        """Applies the two-qubit Copula mixer."""
        for i in range(len(p1) - 1):
            phi1 = 2 * np.arcsin(np.sqrt(p1[i]))
            phi2 = 2 * np.arcsin(np.sqrt(p2[i]))
            qc.ry(phi1, i)
            qc.ry(phi2, i + 1)
            qc.cz(i, i + 1)
            qc.ry(-phi1, i)
            qc.ry(-phi2, i + 1)
            qc.rz(-2 * beta, i)
            qc.rz(-2 * beta, i + 1)


    def QKP(self, betas, gammas, k, theta=None, bit_mapping='regular',
            cost_hamiltonian=False, run_single_job=False, shots=5000):
        """Quantum Knapsack Solver. Simulation mode.

        Args:
            betas (list): Mixer parameter
            gammas (list): Cost parameter
            k (int): State distribution fit parameter
            theta (int, optional): _description_. Defaults to None.
            bit_mapping (str, optional): 'regular' or 'inverse' mapping of the bitstring.
            cost_hamiltonian (bool, optional): _description_. Defaults to False.
            run_single_job (bool, optional): _description_. Defaults to False.
            shots (int, optional): Defaults to 5000.

        Returns:
            _type_: _description_
        """
        
        p = self.logistic_bias(k) # state distribution from greedy solution
        qc = QuantumCircuit(self.n)

        # Initial state preparation
        for i in range(self.n):
            angle = 2 * np.arcsin(np.sqrt(p[i]))
            qc.ry(angle, i)
        qc.barrier()

        for i in range(self.p):

            if cost_hamiltonian:
                self.apply_cost_hamiltonian(qc, gammas[i])

            self.apply_cost_unitary(qc, gammas[i])
            qc.barrier()

            # Mixer application
            if self.mixer == 'X':
                self.apply_X_mixer(qc, betas[i])

            elif self.mixer == 'hourglass':
                self.apply_hourglass_mixer(qc, betas[i], p)

            elif self.mixer == 'copula':
                p2 = self.logistic_bias(k)
                self.apply_copula_mixer(qc, betas[i], p, p2, theta)

        qc.measure_all()



        # Run on real hardware
        if self.run_hardware:
            # Transpilation
            isa_qc = self.pass_manager.run(qc)
            backend = self.backend
            # print(isa_qc)

            if self.generate_jobs == False or run_single_job:
                job = self.sampler.run([isa_qc], shots=shots)
                result = job.result()[0]
            else:
                self.list_transpile_qc.append(isa_qc)
                return None, None, None, None, None

        # Run on local simulator
        else:
            job = self.sampler.run([qc], shots=shots)
            result = job.result()[0]


        counts = result.data.meas.get_counts()

        # Find best solution
        best_solution = max(counts, key=counts.get)

        if bit_mapping == 'regular':
            best_value = sum(int(best_solution[i]) * self.v[i] for i in range(self.n))
            total_weight = sum(int(best_solution[i]) * self.w[i] for i in range(self.n))
        
        if bit_mapping == 'inverse':
            best_value = sum((1 - int(best_solution[i])) * self.v[i] for i in range(self.n))
            total_weight = sum((1 - int(best_solution[i])) * self.w[i] for i in range(self.n))


        # Save parameters results of each runs
        self.dict_all_parameters[f"{betas},{gammas}"] = best_value

        # Check if the solution is valid
        valid = total_weight <= self.c

        return best_solution, float(best_value), total_weight, counts, valid
        # return best_solution, float(best_value), total_weight, counts, valid, qc



    def QKP_value_wrapper(self, betas, gammas, k, theta, bit_mapping, shots):
        """Wrapper that tracks the best bitstring while returning only the value."""

        bitstring, value, weight, counts, valid = self.QKP(betas, gammas, k, theta, bit_mapping, shots=shots)

        if self.generate_jobs == False:
            if (value > self.best_value) and (valid==True):
                print(f"New best solution: {int(value)} -- [{betas}, {gammas}]")
                self.best_bitstring = bitstring
                self.best_value = value
                self.best_weight = weight
                self.best_params = (betas, gammas)
            return value
        


    def grid_search(self, k, theta, N_beta, N_gamma, bit_mapping, shots, show_progress=True):
        """
        Grid search for optimization of β and γ, used for simulation.

        Args:
            k (int): The value of k for the search.
            theta (float): The value of theta for the search.
            N_beta (int): Number of grid points for beta.
            N_gamma (int): Number of grid points for gamma.
            bit_mapping (str): Bit mapping strategy.
            shots (int): Number of shots for simulation.
            show_progress (bool): Whether to display a progress bar.

        Returns:
            tuple: The best beta, gamma, and their corresponding value.
        """
        best_value = -np.inf
        found_opt_sol = False

        beta_values = [np.pi * i / N_beta for i in range(N_beta)]
        gamma_values = [2 * np.pi * j / N_gamma for j in range(N_gamma)]

        # Use tqdm for progress tracking if show_progress is enabled
        beta_iterator = tqdm(product(beta_values, repeat=self.p), desc="Grid Search β", disable=not show_progress)

        for betas in beta_iterator:
            for gammas_combo in product(gamma_values, repeat=self.p):
                if not found_opt_sol:
                    # print('(BETAS, GAMMAS)', betas, gammas_combo)
                    value = self.QKP_value_wrapper(betas, gammas_combo,
                                                   k, theta,
                                                   bit_mapping, shots=shots)

                    if value > best_value:
                        best_value = value
                        best_beta, best_gamma = betas, gammas_combo

                    # Early stopping if optimal solution is found
                    if (self.best_value == self.optimal_solution and 
                        self.speedup_computations):
                        print("Found optimal solution")
                        found_opt_sol = True
                        break

        return best_beta, best_gamma, best_value



    def parameter_optimization(self, k_range, theta_range,
                               N_beta=50, N_gamma=50,
                               bit_mapping='regular', shots=5000):
        """Perform grid search to optimize the different values of betas and gammas."""
        best_value = -np.inf
        
        for k in k_range:
            for theta in theta_range:
                print(f"Parameters (k, θ): {int(k), theta}")
                
                # Grid search
                beta0, gamma0, value = self.grid_search(k, theta, N_beta, N_gamma, bit_mapping, shots=shots)

    
    def generate_circuits(self, k_range, theta_range, beta_values, gamma_values):
        """Generate a list of quantum circuits to solve Knapsack with different parameters.

        Args:
            k_range (_type_): _description_
            theta_range (_type_): _description_
            beta_values (_type_): _description_
            gamma_values (_type_): _description_

        Returns:
            list: List of Qiskit quantum circuits
        """

        list_qc = []

        for k in k_range:
            for theta in theta_range:

                for beta in beta_values:
                    for gamma in gamma_values:
                                
                        p = self.logistic_bias(k)
                        qc = QuantumCircuit(self.n)

                        # Initial state preparation
                        for i in range(self.n):
                            angle = 2 * np.arcsin(np.sqrt(p[i]))
                            qc.ry(angle, i)
                        qc.barrier()

                        # Cost unitary
                        self.apply_cost_unitary(qc, gamma)
                        qc.barrier()

                        # Mixer application
                        if self.mixer == 'X':
                            self.apply_X_mixer(qc, beta)
                        elif self.mixer == 'hourglass':
                            self.apply_hourglass_mixer(qc, beta, p)
                        elif self.mixer == 'copula' and theta is not None:
                            p2 = self.logistic_bias(k)
                            self.apply_copula_mixer(qc, beta, p, p2, theta)
                            
                        qc.measure_all()
                        list_qc.append(qc)

        return list_qc
    

    def transpile_circuits(self, list_qc, pass_manager, show_progess_bar=True):
        """ Transpile a list of quantum circuits.

        Args:
            list_qc (list): List of quantum circuit
            pass_manager: Qiskit pass manager for transpilation
            show_progess_bar (bool, optional): Option to show a progess bar. Defaults to True.

        Returns:
            list: List of transpiled quantum circuit
        """
        list_isa_qc = []
        iterator = tqdm(list_qc, desc="Transpiling circuits") if show_progess_bar else list_qc

        for qc in iterator:
            isa_qc = pass_manager.run(qc)
            list_isa_qc.append(isa_qc)

        return list_isa_qc