import os
import sys
import numpy as np
import Controller
import Reference_States

try:
    # Prefer package-relative imports when available
    from . import Vehicle_Params  # type: ignore
except Exception:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import Vehicle_Params  # type: ignore
    

def init_state_matrices():
    A = np.array([[0, Vehicle_Params.V_x], [0, 0]])
    B = np.array([[0], [Vehicle_Params.V_x / Vehicle_Params.whlb]])
    E = np.array([[0], [- Vehicle_Params.V_x]])
    return A, B, E

def init_state():
    # Initial state: [lateral error, heading error]
    return np.array([[0], [0]])


# Main function to demonstrate usage
def main():
    print("Running Simulation main()\n")

    u, path, curvature, radius, heading = Reference_States.init_path()

    A, B, E = init_state_matrices()
    x = init_state()

    dt = 1 / Vehicle_Params.steering_sampling_rate
    n_steps = len(path[:, 0])  # Simulate for the length of the path

    for k in range(n_steps):
        # For now, use path index directly before adding nearest-point logic
        k_ref = min(k, len(path) - 1)

        kappa_ref = curvature[k_ref]
        psi_ref = heading[k_ref]

        e_y = x[0, 0]
        e_psi = x[1, 0]

        delta = Controller.total_control(e_y, e_psi, kappa_ref)

        x_dot = A @ x + B * delta + E * kappa_ref
        x = x + dt * x_dot


if __name__ == '__main__':
    main()