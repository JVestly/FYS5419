import numpy as np

eps = 1.0
W = 0.2
V = 1.0

H1_1x1 = np.array([[0.0]])
H1_2x2 = np.array([
    [-eps, -V],
    [-V, eps]
])

H2_2x2 = np.array([
    [-eps + 3*W, 3*V],
    [3*V, eps + 3*W]
])

H2_3x3 = np.array([
    [-2*eps, np.sqrt(6)*V, 0],
    [np.sqrt(6)*V, 4*W, np.sqrt(6)*V],
    [0, np.sqrt(6)*V, 2*eps]
])

evals_J1 = np.sort(np.concatenate([np.linalg.eigvalsh(H1_1x1), np.linalg.eigvalsh(H1_2x2)]))
D_J1 = np.diag(evals_J1)

evals_J2 = np.sort(np.concatenate([np.linalg.eigvalsh(H2_2x2), np.linalg.eigvalsh(H2_3x3)]))
D_J2 = np.diag(evals_J2)

np.set_printoptions(precision=4, suppress=True)

print("Diagonalized J=1 (3x3) Matrix:")
print(D_J1)

print("\nDiagonalized J=2 (5x5) Matrix:")
print(D_J2)