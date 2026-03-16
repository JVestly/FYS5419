import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from collections import Counter

#random seed used for measurement measurment 
np.random.seed(42)


def one_qubit_basis():
    #One qubits basis states, computational basis 
    zero = np.array([1,0])
    one = np.array([0,1])
    return zero, one


zero, one = one_qubit_basis()

#Pauli matrices 
I = np.eye(2) #identity 
Z = np.array([[1.0, 0.0], [0.0, -1.0]]) #Pauli Z
X = np.array([[0.0, 1.0], [1.0, 0.0]]) #Pauli X
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]]) #Pauli Y 

#Hadamard and Phase 
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]]) #Hadamard 
S = np.array([[1, 0], [0, 1j]]) #Phase 


#put all gates in list 
gates = [("I",I), ("Z",Z), ("X",X), ("Y",Y), ("Hadamard, H", H), ("Phase gate", S)]

#Function for applying gate to a state 
def apply_gate(gate, state):
    return gate @ state 

#Format digits
def format_state(state):
    return np.round(state, 3)


#collect results of applying gates  
data = []


def apply_multiple_gates(gates):
    #loop through the gates and apply to comp.basis for one qubit
    for name, gate in gates: 
        #apply gate 
        result_zero = apply_gate(gate, zero)
        result_one = apply_gate(gate, one)
        
        #append result to data
        data.append({"Gate": name, 
                    "Gate @ |0> = [1, 0]": format_state(result_zero),
                    "Gate @ |1> = [0, 1]": format_state(result_one)})


    #print a nice table using pandas 
    df = pd.DataFrame(data)

    # Improve display formatting
    pd.set_option("display.max_colwidth", None)

    print(df.to_string(index=False))


def bell_states():
    #Bell states 
    Phi_plus = (1/np.sqrt(2)) * (np.kron(zero, zero) + np.kron(one, one)) 
    Phi_min = (1/np.sqrt(2)) * (np.kron(zero, zero) - np.kron(one, one))
    Psi_plus = (1/np.sqrt(2)) * (np.kron(zero, one) + np.kron(one, zero))
    Psi_min = (1/np.sqrt(2)) * (np.kron(zero, one) - np.kron(one, zero))

    return Phi_plus, Phi_min, Psi_plus, Psi_min

Phi_plus, Phi_min, Psi_plus, Psi_min = bell_states()

#One qubit projectors 
P0 = np.outer(zero, zero) #|0><0|
P1 = np.outer(one, one) # |1><1| 



def cnot_operator(control):
    # Build CNOT with control qubit index (0 or 1)
    if control == 0:
        return np.kron(P0, I) + np.kron(P1, X)
    elif control == 1:
        return np.kron(I, P0) + np.kron(X, P1)
    else:
        raise ValueError("control must be 0 or 1")




def hadamard_which_qubit(qubit_index): 
    if qubit_index == 0: 
        H_on_two = np.kron(H, I)
    else:
        H_on_two = np.kron(I, H)
    return H_on_two


#apply Hadamard followed by a CNOT on Bell state
def Hadamamard_and_CNOT(qubit_index_H, qubit_index_CNOT, state): 
    """
    - qubit_index: index for which qubit to apply Hadamard 
    - param state: the quantum state 
    """
    H = hadamard_which_qubit(qubit_index_H)
    H_state = H @ state
    result = cnot_operator(qubit_index_CNOT) @ H_state
    return result





def measurement(state):
    """
    Measurement in computational basis 
    Arguments: - state, state to be measured 
    """

    #probabilities for qubit0 
    probs0 = np.array([np.sum(np.abs(state[[0,1]])**2), np.sum(np.abs(state[[2,3]])**2)])
    outcome0 = np.random.choice([0,1], p=probs0)
    if outcome0 == 0:
        collapsed = state.copy()
        collapsed[2:] = 0  #amplitudes set to zero 
    else: 
        collapsed = state.copy()
        collapsed[0:2] = 0 

    #normalize 
    norm = np.linalg.norm(collapsed)
    collapsed = collapsed/norm

    #probabilities for qubit 1
    probs1 = np.array([np.sum(np.abs(collapsed[[0,2]])**2), np.sum(np.abs(collapsed[[1,3]])**2)])
    outcome1 = np.random.choice([0,1], p = probs1)

    return outcome0, outcome1 

        
#created with chatGPT for getting statistics and plotting them 
def simulate_measurements(state, n_shots=10000):
    """
    Measure both qubits sequentially n_shots times.
    Returns:
        - counts dictionary
        - empirical probabilities
        - mean and std per qubit
    """
    outcomes_q0 = []
    outcomes_q1 = []
    
    for _ in range(n_shots):
        o0, o1 = measurement(state)
        outcomes_q0.append(o0)
        outcomes_q1.append(o1)
    
    # Counts for each joint outcome
    joint_counts = Counter([f"{o0}{o1}" for o0,o1 in zip(outcomes_q0, outcomes_q1)])
    
    # Empirical probabilities
    joint_probs = {k: v/n_shots for k,v in joint_counts.items()}
    
    # Mean and standard deviation for each qubit
    mean_q0 = np.mean(outcomes_q0)
    std_q0  = np.std(outcomes_q0)
    
    mean_q1 = np.mean(outcomes_q1)
    std_q1  = np.std(outcomes_q1)
    
    return {
        "joint_counts": joint_counts,
        "joint_probs": joint_probs,
        "mean_q0": mean_q0,
        "std_q0": std_q0,
        "mean_q1": mean_q1,
        "std_q1": std_q1
    }





# Suppose Phi_plus is your initial state
final_state = Hadamamard_and_CNOT(1, 1, Psi_min)  # Apply H on qubit 0, CNOT with control 0
results = simulate_measurements(final_state, n_shots=100000)

print("Joint counts:", results["joint_counts"])
print("Joint probabilities:", results["joint_probs"])
print(f"Qubit 0: mean={results['mean_q0']:.9f}, std={results['std_q0']:.3f}")
print(f"Qubit 1: mean={results['mean_q1']:.9f}, std={results['std_q1']:.3f}")
amps = final_state
probs_q0 = [np.sum(np.abs(amps[[0,1]])**2), np.sum(np.abs(amps[[2,3]])**2)]
probs_q1 = [np.sum(np.abs(amps[[0,2]])**2), np.sum(np.abs(amps[[1,3]])**2)]



#create density matrix: 
def create_density(state):
    return np.outer(state, np.conj(state))

#trace out second qubit 
def partial_trace(rho, dims, subsystem):
    """
    Compute partial trace over subsystem
    Arguments - rho, density matrix 
              - dims, tuple of dimensions of subsystem A and B 
              - subsystem, which subsystem to trace out, 0 traces out qubit 0, 1 traces out qubit 1

    Returns:  - returns reduced density matrix 
    """
    dA, dB = dims #dimensions of subsystem A and B 
    rho_reshaped = rho.reshape(dA, dB, dA, dB) #reshape rho 

    if subsystem == 0: 
        #Trace out A, keep B 
        rho_B = np.einsum('ikjk->ij', rho_reshaped) #sum over A axes 
        return rho_B
    elif subsystem == 1: 
        #trace out B, keep A
        rho_A = np.einsum('ijik->jk', rho_reshaped) #sum over B axes 
        return rho_A
    

rho = create_density(Phi_plus)
dims = (2,2) #dimensions of subsystem A and B 
rho_B = partial_trace(rho, dims, 0)
rho_A = partial_trace(rho, dims, 1)

def von_neumann_entropy(rho):
    #obtain eigenvalues of density matrix 
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12] #avoid 0*log(0), when lambda_i = 0, no contribution to entropy. 
    S = -np.sum(eigenvalues * np.log2(eigenvalues))
    #compute von neumann entropy 
    return S

