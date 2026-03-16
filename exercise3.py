import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_algorithms import VQE
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_algorithms.optimizers import ADAM
import warnings
warnings.filterwarnings('ignore')
import exercise1 as e1
import exercise2 as e2
import exercise4 as e4

# Set random seed for reproducibility
np.random.seed(42)





"""
VQE plan: 
1. Ansatz. 

2. Measure E(theta) = <psi(theta)|H|psi(theta)> 


3.optimize, find theta that minimizes E(theta)

4. result ground state energy and wavefunction 
"""


#Gates 
X = e1.X 
Z = e1.Z 
I = e1.I 
H = e1.H
S = e1.S 


#parameters 


#Hamiltonian, H = H0 + lambda*HI. H_I =  cI + w_z Z + w_x X 
c = (e2.V11 + e2.V22)/2
w_z = (e2.V11 - e2.V22)/2 
w_x = e2.V12
eps = (e2.E1+e2.E2)/2
omega = (e2.E1-e2.E2)/2


H_0 = eps*I + omega*Z 
H_I = c*I + w_z*Z + w_x*X



#defining quantum rotation gates 
def Rx(theta): 
    "rotation about x-axis of angle theta"
    Rx = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                   [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
    return Rx

def Ry(theta): 
    "Rotation about y-axis of angle theta"
    Ry = np.array([[np.cos(theta/2), -np.sin(theta/2)], 
                   [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
    return Ry

def Rz(theta): 
    "Rotation about z-axis of angle theta"
    Rz = np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=complex)

    return Rz

def ansatz_state(angles): 
    """
    Ansatz state 
    """
    U = Rz(angles[2])@Ry(angles[1])@Rz(angles[0]) #standard ansatz U = RzRyRz, I shall try with RxRyRx as well 
    state = U @ np.array([1,0])
    return state

def measure_Z(state, shots): 
    """
    Measures state in Z basis 
    Arguments: - State, state to be measured, np.array
               - shots, number of shots (measurements)
    """
    
    #probabilities 
    probs = np.abs(state)**2
   

    #Do measurements and count the number of times we measure 0 and 1
    n0 = np.random.binomial(shots, probs[0])
    n1 = shots - n0

    mean = (n0 - n1)/shots
    counts = (n0, n1)

    return mean, counts


def measure_X(state, shots): 
    """
    Rotate X to Z with Hadamard to measure in X 
    Arguments: - state, state to be measured 
               - shots, number of measurements/shots 
    """

    #rotate state 
    rotated_state = H @ state 

    #perform measurment in Z basis after rotation 
    mean, counts = measure_Z(rotated_state, shots)

    return mean, counts 

def measure_Y(state, shots): 
    """
    Rotate Y to Z with Hadamard gate and dagger of phase gate to measure in Y
    Arguments: - state, state to be measured 
               - shots, number of measurements/shots 
    """

    #rotate state with phase gate and hadamard gate 
    rotated_state = H @ (S.conj().T @ state) 

    #perform measurement in Z basis after rotation 

    mean, counts = measure_Z(rotated_state, shots)

    return mean, counts 



def measure_energy(state, shots, coeffs): 
    """
    Measure energy of the Hamiltonian when it is written as sum of Pauli gates
    Arguments: - state, state to be measured 
               - shots, number of shots 
               - coeffs, dictionary containing the coefficients infront the Pauli-terms in Hamiltonian
    """

    #measure I 
    E_I = coeffs["I"] 

    #measure Z 
    measure_z, _ = measure_Z(state, shots)
    E_Z = coeffs["Z"] * measure_z

    #Measure X 
    measure_x, _ = measure_X(state, shots)
    E_X = coeffs["X"] * measure_x

    E = E_I + E_Z + E_X

    return E 




def gradient_energy(params, shots, coeffs):
    """
    Computes the gradient of the energy with parameter-shift rule 
    Arguments: - params, angles used for the rotation gates
               - shots, number of shots
               - coeffs, coefficients infront of Pauli gates in Hamiltonian
    """
    #angle used in parameter-shift 
    shift = np.pi/2 


    gradients = []
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += shift 

        params_min = params.copy()
        params_min[i] -= shift
        
        #states
        state_plus = ansatz_state(params_plus)
        state_min = ansatz_state(params_min)

        #energies 
        E_plus = measure_energy(state_plus, shots, coeffs)
        E_min = measure_energy(state_min, shots, coeffs)


        #compute gradient
        gradients.append((E_plus - E_min)/2)

    return gradients

class AdamOptimizerVQE: 
    """
    Minimal gradient descent used in VQE 
    """
    def __init__(self, shots, coeffs):
        self.shots = shots 
        self.coeffs = coeffs 

    #compute energy
    def energy(self, params): 
        state = ansatz_state(params)
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
    def optimize_Adam(self, params, lr=0.1, beta_1 = 0.9, beta_2 = 0.99, eps = 1e-9, maxiter = 500):
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

    


    


def find_lowest_eigenvalues():
    params = np.random.uniform(0, 2*np.pi, 3)
    shots = 100

    #Parameter Lambda infront of H_I
    lmbda = np.linspace(0,1,500)
    energies = []
    counter = 0
    for lam in lmbda:
        print(counter)
        #coefficients infront of Pauli matrices, for H_0 and H_1 
        coeffs_both = {"I": eps+c*lam, "Z": omega + w_z*lam, "X":w_x*lam}


        vqe = AdamOptimizerVQE(shots, coeffs_both)
        values = vqe.optimize_Adam(params)

        energies.append(values["energy"])
        counter += 1
        

    plt.plot(lmbda, energies)
    plt.xlabel("λ")
    plt.ylabel("Ground state energy")
    plt.show()

        


#Qiskit version 

def compare_custom_and_qiskit_e3(N=100):
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorEstimator as Estimator
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import ADAM
    import exercise2 as e2
    import numpy as np
    import matplotlib.pyplot as plt

    c = (e2.V11 + e2.V22) / 2
    w_z = (e2.V11 - e2.V22) / 2 
    w_x = e2.V12
    eps = (e2.E1 + e2.E2) / 2
    omega = (e2.E1 - e2.E2) / 2

    params_vec = ParameterVector("theta", 3)
    qc = QuantumCircuit(1)
    qc.rz(params_vec[0], 0)
    qc.ry(params_vec[1], 0)
    qc.rz(params_vec[2], 0)

    estimator = Estimator()
    qiskit_optimizer = ADAM(maxiter=500, lr=0.1)

    lmbda = np.linspace(0, 1, N)
    custom_energies = []
    qiskit_energies = []
    

    init_angles = np.random.uniform(0, 2*np.pi, 3)
    
    for count, lam in enumerate(lmbda):
        if count % 10 == 0:
            print(f"Comparing e3: {count}/{N}")
            
  
        coeffs_both = {"I": eps + c*lam, "Z": omega + w_z*lam, "X": w_x*lam}
        

        vqe_custom = AdamOptimizerVQE(shots=5000, coeffs=coeffs_both)
        res_custom = vqe_custom.optimize_Adam(init_angles)
        custom_energies.append(res_custom["energy"])
        

        H = SparsePauliOp.from_list([
            ("I", coeffs_both["I"]), 
            ("Z", coeffs_both["Z"]), 
            ("X", coeffs_both["X"])
        ])
        
        vqe_q = VQE(estimator, qc, qiskit_optimizer, initial_point=init_angles)
        res_q = vqe_q.compute_minimum_eigenvalue(H)
        qiskit_energies.append(res_q.eigenvalue.real)


    plt.figure(figsize=(8, 5))
    plt.plot(lmbda, custom_energies, 'b-', label="Custom VQE", alpha=0.7)
    plt.plot(lmbda, qiskit_energies, 'r--', label="Qiskit VQE", alpha=0.7)
    
    plt.xlabel("λ")
    plt.ylabel("Ground State Energy")
    plt.title("Comparison: Custom vs Qiskit VQE (1-Qubit)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig("vqe_comparison_e3.pdf", bbox_inches="tight")
    print("Saved: vqe_comparison_e3.pdf")

if __name__ == "__main__":
    compare_custom_and_qiskit_e3(10)