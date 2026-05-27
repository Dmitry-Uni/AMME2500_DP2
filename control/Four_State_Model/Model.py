import control.Vehicle_Params as Vehicle_Params
import numpy as np

def build_state_matrices():

    Vx = Vehicle_Params.V_x
    m = Vehicle_Params.mass
    Cf = Vehicle_Params.Cf
    Cr = Vehicle_Params.Cr
    lf = Vehicle_Params.lf
    lr = Vehicle_Params.lr
    Iz = Vehicle_Params.yaw_inertia

    A = np.array([
        [0.0, Vx, 1.0, 0.0],

        [0.0, 0.0, 0.0, 1.0],

        [0.0, 0.0,
         -(2 * Cf + 2 * Cr) / (m * Vx),
         ((-2 * Cf * lf + 2 * Cr * lr) / (m * Vx)) - Vx],

        [0.0, 0.0,
         (-2 * Cf * lf + 2 * Cr * lr) / (Iz * Vx),
         -(2 * Cf * lf**2 + 2 * Cr * lr**2) / (Iz * Vx)]
    ])

    B = np.array([
        [0.0],
        [0.0],
        [2 * Cf / m],
        [2 * Cf * lf / Iz]
    ])

    E = np.array([
        [0.0],
        [- Vx],
        [0.0],
        [0.0]
    ])

    return A, B, E

def controllability_matrix(A: np.ndarray, B: np.ndarray):
    n = A.shape[0]
    controllability_matrix = B
    for i in range(1, n):
        controllability_matrix = np.hstack((controllability_matrix, np.linalg.matrix_power(A, i) @ B))
    return controllability_matrix

def check_stability(A: np.ndarray):
    eigenvalues = np.linalg.eigvals(A)
    if np.all(np.real(eigenvalues) < 0):
        print("The system is stable.")
    else:
        print("The system is NOT stable.")
    return np.all(np.real(eigenvalues) < 0)

def check_controllability(A: np.ndarray, B: np.ndarray):
    n = A.shape[0]
    cm = controllability_matrix(A, B)
    rank = np.linalg.matrix_rank(cm)
    if rank == n:
        print(f"The system is controllable. {rank == n} (Rank: {rank}, Size: {n})")
    else:
        print(f"The system is NOT controllable. {rank == n} (Rank: {rank}, Size: {n})")
    return rank == n

def check_open_loop_modes(A: np.ndarray):
    full_eigs = np.linalg.eigvals(A)
    dyn_eigs = np.linalg.eigvals(A[2:4, 2:4])

    print(f"Full A eigenvalues: {full_eigs}")
    print(f"Lateral/yaw subsystem eigenvalues: {dyn_eigs}")

    full_asymptotically_stable = np.all(np.real(full_eigs) < 0)
    full_marginal = np.all(np.real(full_eigs) <= 1e-9)

    dyn_stable = np.all(np.real(dyn_eigs) < 0)

    print(f"Full system asymptotically stable: {full_asymptotically_stable}")
    print(f"Full system marginally stable: {full_marginal}")
    print(f"Lateral/yaw subsystem stable: {dyn_stable}")

    return full_eigs, dyn_eigs


def main():
    A, B, E = build_state_matrices()
    check_controllability(A, B)
    check_open_loop_modes(A)

if __name__ == "__main__":
    main()