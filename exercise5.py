import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_algorithms import VQE
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_algorithms.optimizers import ADAM
warnings.filterwarnings('ignore')
import exercise1 as e1
import exercise3 as e3
import exercise4 as e4


"""
Here we must have Hamiltonian coefficients as well, but I dont understand the lambda part of exercise e) 
"""


def ansatsz(params):
    """
    Prepares ansatz state for VQE algorithm
    
    Arguments - params: parameters used for the rotation gates in the ansatz 
    """
    #First layer of rotations in circuit, trying with RxRz first, might try something different later 
    U1_0 = e3.Rx(params[0]) @ e3.Rz(params[1])
    U1_1 = e3.Rx(params[2]) @ e3.Rz(params[3])
    U1 = np.kron(U1_0, U1_1)

    #Second layer (after CNOT)
    U2_0 = e3.Rz(params[4]) @ e3.Rx(params[5])
    U2_1 = e3.Rz(params[6]) @ e3.Rx(params[7])
    U2 = np.kron(U2_0, U2_1)

    #squeeze CNOT between layers, first qubit is control 
    U = U2 @ e1.cnot_operator(control=0) @ U1

    #return U applied to |00> 
    return U @ np.array([1,0,0,0])



def measurement_two_qubits(state, shots): 
    """
    Perform measurement on the two qubits simultaneously in computational basis
    Counts the number of times each outcome occurs 

    Arguments: - state, state to be measured
               - shots, number of shots 
    """
    #probabilities 
    probs = np.abs(state)**2

    measurements = np.random.choice(4, size = shots, p = probs)


    counts = {"00":np.sum(measurements==0), "01":np.sum(measurements==1), "10":np.sum(measurements==2), "11":np.sum(measurements==3)}
    return counts 

def measure_ZZ(state, shots):
    """
    Measure ZZ in comp basis 
    Arguments: - state, state to be measured 
               - shots, number of shots 
    """
    counts = measurement_two_qubits(state, shots)
    mean = (counts["00"] + counts["11"] - counts["01"] - counts["10"])/shots

    return mean, counts

def measure_IZ(state, shots):
    """
    Measure IZ in comp basis 
    Arguments: - state, state to be measured 
               - shots, number of shots 
    """
    counts = measurement_two_qubits(state, shots)
    mean = (counts["00"] + counts["10"] - counts["01"] - counts["11"])/shots

    return mean, counts 

def measure_ZI(state, shots):
    """
    Measure ZI in comp basis 
    Arguments: - state, state to be measured 
               - shots, number of shots 
    """
    counts = measurement_two_qubits(state, shots)
    mean = (counts["00"] + counts["01"] - counts["10"] - counts["11"])/shots

    return mean, counts 

def measure_XX(state, shots):
    """
    Measure XX in comp basis, rotate to XX to ZZ then measure 
    Arguments: - state, state to be measured 
               - shots, number of shots 
    """
    H_H = np.kron(e3.H, e3.H) #rotate 
    rotated_state = H_H @ state
    mean, counts = measure_ZZ(rotated_state, shots)
   
    return mean, counts 

def measure_energy(state, shots, coeffs):
    """
    Measures total energy 
    Arguments: - state, state to be measured 
               - shots, number of shots 
               - coeffs, coefficients infront of Pauli terms of Hamiltonian 
    """    
    #II term 
    E_II = coeffs["I"]

    #IZ term
    e_iz, _ = measure_IZ(state, shots)
    E_IZ = coeffs["IZ"] * e_iz

    #ZI term 
    e_zi, _ = measure_ZI(state, shots)
    E_ZI = coeffs["ZI"] * e_zi

    #ZZ term 
    e_zz, _ = measure_ZZ(state, shots)
    E_ZZ  = coeffs["ZZ"] * e_zz

    #XX term 
    e_xx, _ = measure_XX(state, shots)
    E_XX = coeffs["XX"] * e_xx

    #total energy
    E_tot = E_II + E_IZ + E_ZI + E_ZZ + E_XX

    return E_tot



class AdamOptimizerVQE_two_qubit: 
    """
    Minimal Adam optimizer used in VQE 
    """
    def __init__(self, shots, coeffs):
        self.shots = shots 
        self.coeffs = coeffs 

    #compute energy
    def energy(self, params): 
        state = ansatsz(params)
        return measure_energy(state, self.shots, self.coeffs)
    
    #compute gradient using parameter-shift rule 
    def gradient(self, params, shift = np.pi/2):
        params = np.asarray(params, dtype=float)
        grads = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_min = params.copy()
            p_plus[i] += shift 
            p_min[i] -= shift 
            e_plus = self.energy(p_plus)
            e_min = self.energy(p_min)
            grads[i] = (e_plus-e_min)/2
        return grads
    
    #Optimiziation part, Adam Optimizer 
    def optimize_Adam(self, params, lr=0.01, beta_1 = 0.9, beta_2 = 0.99, eps = 1e-9, maxiter = 500):
        angles = np.asarray(params, dtype=float).copy()
        m = np.zeros_like(angles)
        v = np.zeros_like(angles)
        history = []
        for i in range(1, maxiter + 1):
            g = self.gradient(angles)
            m = beta_1 * m + (1 - beta_1) * g
            v = beta_2 * v + (1 - beta_2) * (g * g)
            m_hat = m/(1-beta_1**i)
            v_hat = v/(1-beta_2**i)
            angles = angles - lr * m_hat/(np.sqrt(v_hat) + eps)
            history.append(self.energy(angles))

        return {"angles":angles, "energy":history[-1]}


def find_lowest_eigenvalues(N):
    """
    Find and plot lowest eigenvalue as function of lambda
    Arguments - N, number of different lambda values
    """
    params = np.random.uniform(0, 2*np.pi, 8)
    shots = 500

    #Parameter Lambda infront of H_I
    lmbda = np.linspace(0, 1, N)
    energies = []
    counter = 0
    for lam in lmbda:
        print(counter)
        #coefficients infront of Pauli matrices, for H_0 and H_1 
        _, coeffs = e4.construct_Hamiltonian(e4.eps_00, e4.eps_11, e4.eps_22, e4.eps_33, e4.Hx, e4.Hz, lam)

        print(coeffs)
        vqe = AdamOptimizerVQE_two_qubit(shots, coeffs)
        values = vqe.optimize_Adam(params)

        energies.append(values["energy"])
        counter += 1
        

    plt.plot(lmbda, energies)
    plt.xlabel("λ")
    plt.ylabel("Ground state energy")
    plt.show()



"""
Maybe some code for drawing the circuit here 
"""

def compare_custom_and_qiskit_e5(N=30):
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorEstimator as Estimator
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import ADAM
    import exercise4 as e4
    import numpy as np
    import matplotlib.pyplot as plt

    # Qiskit Circuit Setup
    params_vec = ParameterVector("theta", 8)
    qc = QuantumCircuit(2)
    qc.rx(params_vec[0], 0)
    qc.rz(params_vec[1], 0)
    qc.rx(params_vec[2], 1)
    qc.rz(params_vec[3], 1)
    qc.cx(0, 1)
    qc.rz(params_vec[4], 0)
    qc.rx(params_vec[5], 0)
    qc.rz(params_vec[6], 1)
    qc.rx(params_vec[7], 1)

    estimator = Estimator()
    qiskit_optimizer = ADAM(maxiter=500, lr=0.1)

    lmbda = np.linspace(0, 1, N)
    custom_energies = []
    qiskit_energies = []
    

    init_angles = np.random.uniform(0, 2*np.pi, 8)
    
    for count, lam in enumerate(lmbda):
        

        _, coeffs = e4.construct_Hamiltonian(e4.eps_00, e4.eps_11, e4.eps_22, e4.eps_33, e4.Hx, e4.Hz, lam)
        

        vqe_custom = AdamOptimizerVQE_two_qubit(shots=5000, coeffs=coeffs)
        val = vqe_custom.optimize_Adam(init_angles)
        custom_energies.append(val["energy"])

        H = SparsePauliOp.from_list([
            ("II", coeffs.get("I", 0.0)),
            ("ZI", coeffs.get("IZ", 0.0)), # Qiskit ZI = Z on q1, I on q0
            ("IZ", coeffs.get("ZI", 0.0)), # Qiskit IZ = I on q1, Z on q0
            ("ZZ", coeffs.get("ZZ", 0.0)),
            ("XX", coeffs.get("XX", 0.0))
        ])
        
        vqe_qiskit = VQE(estimator, qc, qiskit_optimizer, initial_point=init_angles)
        result = vqe_qiskit.compute_minimum_eigenvalue(H)
        qiskit_energies.append(result.eigenvalue.real)

    plt.figure(figsize=(8, 5))
    

    plt.plot(lmbda, custom_energies, label="Custom VQE", color='tab:blue', marker='o', markersize=5, linestyle='-', alpha=0.8)
    
    plt.plot(lmbda, qiskit_energies, label="Qiskit VQE", color='tab:orange', marker='s', markersize=5, linestyle='--', alpha=0.8)
    
    plt.xlabel("Coupling Strength λ")
    plt.ylabel("Ground State Energy")
    plt.title("VQE Assessment: Custom Implementation vs. Qiskit")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig("vqe_comparison_e5.pdf", bbox_inches="tight")
    print("Comparison plot saved successfully as vqe_comparison_e5.pdf!")

if __name__ == "__main__":
    compare_custom_and_qiskit_e5(10)