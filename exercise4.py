import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import exercise1 as e1


#define two-qubit Pauli gates 
II = np.kron(e1.I, e1.I)
IZ = np.kron(e1.I, e1.Z)
ZI = np.kron(e1.Z, e1.I)
ZZ = np.kron(e1.Z, e1.Z)
XX = np.kron(e1.X, e1.X)
YY = np.kron(e1.Y, e1.Y)


#parameters 
eps_00 = 0.0 
eps_11 = 2.5
eps_22 = 6.5 
eps_33 = 7.0
Hx = 2.0
Hz = 3.0


def construct_Hamiltonian(eps_00, eps_11, eps_22, eps_33, Hx, Hz, lam):
    """
    Construct Hamiltonian
    Argument: lam, interaction strength parameter
    """
    #Write Hamiltonian as sum of Pauli operators, coefficients are defined 
    alpha = (1/4)*(eps_00+eps_11+eps_22+eps_33)
    beta = (1/4)*(eps_00+eps_11-eps_22-eps_33)
    gamma = (1/4)*(eps_00-eps_11+eps_22-eps_33)
    delta = (1/4)*(eps_00-eps_11-eps_22+eps_33)
    #Hamiltonian, H0, HI and total
    H_0 = alpha*II + beta*ZI + gamma*IZ + delta*ZZ
    H_I = Hx * XX + Hz * ZZ
    #Full Hamiltonian
    Hamiltonian = H_0 + lam*H_I
    #print(Hamiltonian)
    coeffs = {"I":alpha, "ZI":beta, "IZ":gamma, "ZZ":delta+lam*Hz, "XX":lam*Hx}
    return Hamiltonian, coeffs 

def construct_comp_basis():
    zero_zero = np.array([1,0,0,0])
    zero_one = np.array([0,1,0,0])
    one_zero = np.array([0,0,1,0])
    one_one = np.array([0,0,0,1])
    return zero_zero, zero_one, one_zero, one_one


def compute_eigvals(H):
    """
    Computes the eigenvalues of Hamiltonian 
    
    Argument: -  H, Hamiltonian of system 
    """
    eigvals = np.linalg.eigvalsh(H)
    return eigvals



def multiple_lmbda(N):
    """
    Computes the eigenvalues for different values of parameter strength lmbda 
    
    Arguments: - N, number of different lmbda values to compute eigenvalues for
    """
    lmbda = np.linspace(0,1,N)
    eigvals = np.zeros((N,4)) #empty to hold eigvals later 
    for i, lam in enumerate(lmbda): 
        H, _ = construct_Hamiltonian(eps_00, eps_11, eps_22, eps_33, Hx, Hz, lam)
        eigvals_lam = compute_eigvals(H)
        eigvals[i,:] = eigvals_lam

    return lmbda, eigvals


def compute_entanglement(N, dims, subsystem):
    """
    Computes entanglement for two-qubit system as function of interaction strength lambda

    Arguments: - dims, dimensions of subsystems, tuple
               - subsystem, which subsystem to trace out 
    """
    lmbda = np.linspace(0,1,N)
    entropies = np.zeros_like(lmbda)
    for i,lam in enumerate(lmbda): 
        #construct Hamiltonian and obtain eigenstates 
        H, _= construct_Hamiltonian(lam)
        _, eigvecs = np.linalg.eigh(H)
        #get lowest energy state and compute its density matrix
        lowest_state = eigvecs[:,0] 
        rho = e1.create_density(lowest_state)
        #Take partial trace of qubit 1
        reduced_rho = e1.partial_trace(rho, dims, subsystem = subsystem)
        #Compute Von Neumann entropy 
        entropy = e1.von_neumann_entropy(reduced_rho)
        entropies[i] = entropy
    return lmbda, entropies 



def plot_entropy_vs_lmbda():
    lmbda, entropies = compute_entanglement(N=500, dims=(2,2), subsystem=1)
  

    plt.plot(lmbda, entropies)
    plt.xlabel(r"$\lambda$", fontsize = 14)
    plt.ylabel("Von Neumann entropy", fontsize = 14)
    plt.title("Interaction strengt lambda vs Entropy")
    plt.grid(True)
    plt.show()


def plot_lmbda_vs_eigvals():
    lmbda, eigvals = multiple_lmbda(N=500)

    plt.plot(lmbda, eigvals[:, 0], label="Eigenvalue 1")
    plt.plot(lmbda, eigvals[:, 1], label="Eigenvalue 2")
    plt.plot(lmbda, eigvals[:, 2], label="Eigenvalue 3")
    plt.plot(lmbda, eigvals[:, 3], label="Eigenvalue 4")

    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel("Eigenvalues", fontsize=15)
    plt.title("Eigenvalues vs lambda",fontsize=16)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.show()


plot_lmbda_vs_eigvals()