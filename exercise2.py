import numpy as np 
import matplotlib.pyplot as plt 

# Defining variables and setting up the Hamiltonian

E1 = 0
E2 = 4
V11 = 3
V22 = -V11
V12 = V21 = 0.2



"""
Is it accepted to solve the Hamiltonian H_I when written as such? 
We havent written it in terms of sigma matrices. 
"""

def hamiltonian(l=0.0):
    H_0 = np.array([[E1, 0], [0, E2]])
    H_i = np.array([[V11, V21], [V12, V22]])    
    return np.array(H_0 + l*H_i)

real = [[3, 0.2], [0.2, -3]]

N = 500
lams = np.linspace(0, 1, N)

eigvals = np.zeros((N, 2))

for i, lam in enumerate(lams):
    w, v = np.linalg.eigh(hamiltonian(lam))
    eigvals[i, :] = w

# Plot
plt.plot(lams, eigvals[:, 0], label="Eigenvalue 1")
plt.plot(lams, eigvals[:, 1], label="Eigenvalue 2")

plt.xlabel(r"$\lambda$", fontsize=15)
plt.ylabel("Eigenvalues", fontsize=15)
plt.title("Eigenvalues vs lambda",fontsize=16)
plt.legend(fontsize=13)
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters from the problem
E1 = 0
E2 = 4
V11 = 3
V22 = -3
V12 = 0.2

# Derived parameters
epsilon = (E1 + E2) / 2
Omega = (E1 - E2) / 2
c = (V11 + V22) / 2
omega_z = (V11 - V22) / 2
omega_x = V12

# Pauli matrices
sigma_x = np.array([[0,1],[1,0]])
sigma_z = np.array([[1,0],[0,-1]])
I = np.eye(2)

# Lambda values
lam = np.linspace(0,1,200)

E_plus = []
E_minus = []
E_num1 = []
E_num2 = []

for l in lam:

    # analytic eigenvalues
    term = np.sqrt((Omega + l*omega_z)**2 + (l*omega_x)**2)
    E_plus.append((epsilon + l*c) + term)
    E_minus.append((epsilon + l*c) - term)

    # build Hamiltonian and diagonalize numerically
    H = (epsilon + l*c)*I + (Omega + l*omega_z)*sigma_z + (l*omega_x)*sigma_x
    evals = np.linalg.eigvalsh(H)

    E_num1.append(evals[0])
    E_num2.append(evals[1])

# Plot
plt.plot(lam, E_plus, label="Analytic $E_+$")
plt.plot(lam, E_minus, label="Analytic $E_-$")

plt.plot(lam, E_num1, '--', label="Numeric $E_1$")
plt.plot(lam, E_num2, '--', label="Numeric $E_2$")

plt.xlabel(r"$\lambda$")
plt.ylabel("Energy")
plt.title("Eigenvalues vs interaction strength")
plt.legend()
plt.grid()
plt.show()






import numpy as np
import matplotlib.pyplot as plt

# Parameters
E1 = 0
E2 = 4
V11 = 3
V22 = -3
V12 = 0.2

epsilon = (E1 + E2) / 2
Omega = (E1 - E2) / 2
c = (V11 + V22) / 2
omega_z = (V11 - V22) / 2
omega_x = V12

sigma_x = np.array([[0, 1], [1, 0]], dtype=float)
sigma_z = np.array([[1, 0], [0, -1]], dtype=float)
I = np.eye(2)

lam_vals = np.linspace(0, 1, 400)

# Store |0>- and |1>-weights for lower and upper eigenstates
lower_0 = []
lower_1 = []
upper_0 = []
upper_1 = []

for lam in lam_vals:
    H = (epsilon + lam*c)*I + (Omega + lam*omega_z)*sigma_z + (lam*omega_x)*sigma_x

    # eigvalsh/eigh sorts eigenvalues ascending
    evals, evecs = np.linalg.eigh(H)

    v_lower = evecs[:, 0]
    v_upper = evecs[:, 1]

    lower_0.append(abs(v_lower[0])**2)
    lower_1.append(abs(v_lower[1])**2)
    upper_0.append(abs(v_upper[0])**2)
    upper_1.append(abs(v_upper[1])**2)

plt.plot(lam_vals, lower_0, label=r"Lower state: $|\langle 0|\psi_-\rangle|^2$")
plt.plot(lam_vals, lower_1, label=r"Lower state: $|\langle 1|\psi_-\rangle|^2$")
plt.plot(lam_vals, upper_0, "--", label=r"Upper state: $|\langle 0|\psi_+\rangle|^2$")
plt.plot(lam_vals, upper_1, "--", label=r"Upper state: $|\langle 1|\psi_+\rangle|^2$")

plt.xlabel(r"$\lambda$")
plt.ylabel("Weight")
plt.title("Computational-basis character of eigenstates")
plt.legend()
plt.grid()
plt.show()