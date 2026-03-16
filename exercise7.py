import numpy as np
import matplotlib.pyplot as plt
import time

import numpy as np
import matplotlib.pyplot as plt
import time


# I used the implementation from 3 and 5, but had to recreate the class to cope with different Pauli expressions etc.
# Also, something weird happened with the main threads when trying to call on the class from other scripts. 
class QuasispinVQE:
    def __init__(self, epsilon, V):
        self.epsilon = epsilon
        self.V = V

    def get_expectation(self, theta):
        return -self.epsilon * np.cos(theta) - self.V * np.sin(theta)

    def parameter_shift_gradient(self, theta):
        shift = np.pi / 2
        return (self.get_expectation(theta + shift) - self.get_expectation(theta - shift)) / 2

    def solve(self, mode="ground", lr=0.1, max_iter=200):
        theta = 0.5  
        m, v = 0, 0
        beta1, beta2 = 0.9, 0.999
        eps_stable = 1e-8
        sign = 1 if mode == "ground" else -1

        for i in range(1, max_iter + 1):
            grad = sign * self.parameter_shift_gradient(theta)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**i)
            v_hat = v / (1 - beta2**i)
            theta -= lr * m_hat / (np.sqrt(v_hat) + eps_stable)
            
        return self.get_expectation(theta)

def adam_optimizer(func, init_params, lr=0.1, max_iter=300):
    params = np.array(init_params, dtype=float)
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2 = 0.9, 0.999
    eps_stable = 1e-8
    delta = 1e-5
    
    for i in range(1, max_iter + 1):
        grads = np.zeros_like(params)
        for j in range(len(params)):
            shift = np.zeros_like(params)
            shift[j] = delta
            grads[j] = (func(params + shift) - func(params - shift)) / (2 * delta)
        
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads**2)
        m_hat = m / (1 - beta1**i)
        v_hat = v / (1 - beta2**i)
        
        params -= lr * m_hat / (np.sqrt(v_hat) + eps_stable)
    return params

class QuasispinVQE_J2:
    def __init__(self, eps, V, W):
        self.eps = eps
        self.V = V
        self.W = W

    def h_2x2_energy(self, p):
        theta = p[0]
        z, x = np.cos(theta), np.sin(theta)
        return 3*self.W - self.eps * z + 3*self.V * x

    def h_3x3_energy(self, p):
        a, b = p
        v = np.array([np.cos(a), np.sin(a)*np.cos(b), np.sin(a)*np.sin(b)])
        diag = np.array([-2*self.eps, 4*self.W, 2*self.eps])
        return np.dot(v**2, diag) + 2 * np.sqrt(6)*self.V * (v[0]*v[1] + v[1]*v[2])

    def solve(self):
        results = []
        
        p_opt = adam_optimizer(lambda p: self.h_2x2_energy(p), [0.0])
        results.append(self.h_2x2_energy(p_opt))
        
        p_opt = adam_optimizer(lambda p: -self.h_2x2_energy(p), [0.0])
        results.append(self.h_2x2_energy(p_opt))
        
        p_opt = adam_optimizer(lambda p: self.h_3x3_energy(p), [0.5, 0.5])
        results.append(self.h_3x3_energy(p_opt))
        
        p_opt = adam_optimizer(lambda p: -self.h_3x3_energy(p), [0.5, 0.5])
        results.append(self.h_3x3_energy(p_opt))
        
        p_opt = adam_optimizer(lambda p: self.h_3x3_energy(p)**2, [1.5, 0.5])
        results.append(self.h_3x3_energy(p_opt))
        
        return np.sort(results)

eps_val = 1.0
V_vals = np.linspace(0, 2, 40)
results = {"ground": [], "mid": [], "excited": []}
exact = {"ground": [], "mid": [], "excited": []}
times_3x3 = {"vqe": [], "exact": []}

for V in V_vals:
    vqe = QuasispinVQE(eps_val, V)
    
    start_exact = time.perf_counter()
    H_exact = np.array([[-eps_val, 0, -V], [0, 0, 0], [-V, 0, eps_val]])
    evals = np.sort(np.linalg.eigvalsh(H_exact))
    exact["ground"].append(evals[0])
    exact["mid"].append(evals[1])
    exact["excited"].append(evals[2])
    times_3x3["exact"].append(time.perf_counter() - start_exact)

    start_vqe = time.perf_counter()
    results["ground"].append(vqe.solve(mode="ground"))
    results["mid"].append(0) 
    results["excited"].append(vqe.solve(mode="excited"))
    times_3x3["vqe"].append(time.perf_counter() - start_vqe)

plt.figure(figsize=(6, 4))
plt.plot(V_vals, exact["ground"], 'k-', alpha=0.3, label="Exact")
plt.plot(V_vals, exact["mid"], 'k-', alpha=0.3)
plt.plot(V_vals, exact["excited"], 'k-', alpha=0.3)
plt.scatter(V_vals, results["ground"], color='crimson', s=20, label="VQE Ground")
plt.scatter(V_vals, results["mid"], color='gray', s=20, label="VQE Decoupled")
plt.scatter(V_vals, results["excited"], color='royalblue', s=20, label="VQE Excited")
plt.title("3x3: VQE vs Exact Energies")
plt.xlabel("Coupling Strength V")
plt.ylabel("Energy")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("plot1_3x3_energy.pdf", format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(V_vals, times_3x3["exact"], label="Exact Diag.", color='black')
plt.plot(V_vals, times_3x3["vqe"], label="VQE (Adam)", color='purple')
plt.title("3x3: Execution Time")
plt.xlabel("Coupling Strength V")
plt.ylabel("Time (s)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("plot2_3x3_time.pdf", format="pdf", bbox_inches="tight")
plt.show()

eps, W = 1.0, 0.0
V_vals_5x5 = np.linspace(0.01, 2, 40)
vqe_data, exact_data = [], []
times_5x5 = {"vqe": [], "exact": []}

for v in V_vals_5x5:
    start_vqe = time.perf_counter()
    vqe_data.append(QuasispinVQE_J2(eps, v, W).solve())
    times_5x5["vqe"].append(time.perf_counter() - start_vqe)
    
    start_exact = time.perf_counter()
    H = np.array([
        [-2*eps, 0, np.sqrt(6)*v, 0, 0],
        [0, -eps+3*W, 0, 3*v, 0],
        [np.sqrt(6)*v, 0, 4*W, 0, np.sqrt(6)*v],
        [0, 3*v, 0, eps+3*W, 0],
        [0, 0, np.sqrt(6)*v, 0, 2*eps]
    ])
    exact_data.append(np.sort(np.linalg.eigvalsh(H)))
    times_5x5["exact"].append(time.perf_counter() - start_exact)

vqe_res, ex_res = np.array(vqe_data), np.array(exact_data)
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']

plt.figure(figsize=(6, 4))
for i in range(5):
    plt.plot(V_vals_5x5, ex_res[:, i], 'k-', alpha=0.4, label='Exact' if i==0 else "")
    plt.plot(V_vals_5x5, vqe_res[:, i], 'o', markersize=3, color=colors[i], label=f'VQE E{i}')

plt.xlabel('Coupling Strength V')
plt.ylabel('Energy')
plt.title('5x5: VQE vs Exact Energies')
plt.legend(fontsize=8, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot3_5x5_energy.pdf", format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(V_vals_5x5, times_5x5["exact"], label="Exact Diag.", color='black')
plt.plot(V_vals_5x5, times_5x5["vqe"], label="VQE (Adam)", color='purple')
plt.xlabel("Coupling Strength V")
plt.ylabel("Time (s)")
plt.title("5x5: Execution Time")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("plot4_5x5_time.pdf", format="pdf", bbox_inches="tight")
plt.show()