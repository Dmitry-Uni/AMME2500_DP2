from . import Model
import control.Reference_States as Reference_States
import control.Vehicle_Params as Vehicle_Params
import numpy as np


def init_state(e_y0=0.8, e_psi0=np.radians(3.0), e_y_dot0=0.0, e_psi_dot0=0.0):
    """
    Initial state: [lateral error, heading error]^T.

    Non-zero initial error is useful for report plots because it shows
    convergence back toward the reference path.
    """
    return np.array([
        [e_y0],
        [e_psi0],
        [e_y_dot0],
        [e_psi_dot0]
    ])

def compute_path_arclength(path):
    """
    Compute cumulative arc length along the reference path.
    """
    diffs = np.diff(path, axis=0)
    ds = np.linalg.norm(diffs, axis=1)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    return s

def simulate_open_loop(A, B, E, path, curvature):
    """
    Simulate the open-loop response of the system to a given input and disturbance.

    Parameters:
    - A, B, E: State-space matrices
    - initial_state: Initial state vector
    - u_func: Function of time that returns the control input
    - disturbance_func: Function of time that returns the disturbance
    - t_final: Final simulation time
    - dt: Time step for simulation

    Returns:
    - time: Array of time points
    - states: Array of state vectors at each time point
    """
    s = compute_path_arclength(path)
    t = s / Vehicle_Params.V_x

    x = init_state()
    n_steps = len(path)

    x_history = np.zeros((n_steps, 2))
    delta_history = np.zeros(n_steps)

    x_history[0, :] = x[:, 0]

    states = np.zeros((A.shape[0], n_steps))
    states[:, 0] = x.flatten()

    e_y = x[0, 0]
    e_psi = x[1, 0]
    e_y_dot = x[2, 0]
    e_psi_dot = x[3, 0]

    for i in range(1, n_steps):
        dt = t[i] - t[i - 1]

        kappa_ref = curvature[i]

        t = i * dt
        delta = 0.0  # No control input for open-loop simulation

        x_dot = A @ x + B * delta + E * kappa_ref
        x = x + x_dot * dt

    return t, states

def main():

    #u, path, curvature, radius, heading = Reference_States.init_path()

    A, B, E = Model.build_state_matrices()
    print("State matrix A:\n", A)
    print("Input matrix B:\n", B)
    print("Disturbance matrix E:\n", E)

    Model.check_stability(A)

    Model.check_controllability(A, B)

    Model.check_open_loop_modes(A)


if __name__ == "__main__":
    main()