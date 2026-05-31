from . import Model
from .. import Vehicle_Params
from scipy.signal import place_poles

import numpy as np

def build_controller_matrix(A: np.ndarray, B: np.ndarray, p_K: np.ndarray):
    """
    Build the state feedback gain matrix K for pole placement.

    Parameters:
        A (np.ndarray): State matrix of the system.
        B (np.ndarray): Input matrix of the system.
        p_K (np.ndarray): Desired closed-loop pole locations.

    Returns:
        K (np.ndarray): State feedback gain matrix.
    """

    # Use scipy's place_poles function to compute the gain matrix K
    result = place_poles(A, B, p_K)
    K = result.gain_matrix
    return K

def build_observer_matrix(A: np.ndarray, C: np.ndarray, p_L: np.ndarray):
    """
    Build the observer gain matrix L for pole placement.

    Parameters:
        A (np.ndarray): State matrix of the system.
        C (np.ndarray): Output matrix of the system.
        p_L (np.ndarray): Desired observer pole locations.
    Returns:
        L (np.ndarray): Observer gain matrix.
    """

    # Transpose A and C for observer design (dual system)
    result = place_poles(A.T, C.T, p_L)
    L = result.gain_matrix.T  # Transpose back to get L
    return L

def controller_poles():
    # Desired closed-loop pole locations for the controller
    p_K = np.array([-2.3 + 2.4j, -2.3 - 2.4j, -8.5, -14.0])  # Trial pole locations (real negative)
    return p_K

def observer_poles():
    # Desired observer pole locations
    p_L = np.array([-6 + 6j, -6 - 6j, -15, -18])  # Trial pole locations (real negative)
    return p_L

def controller_eigenvalues(A: np.ndarray, B: np.ndarray, K: np.ndarray):
    # Compute the closed-loop eigenvalues of the system with state feedback
    A_cl = A - B @ K
    eigenvalues = np.linalg.eigvals(A_cl)
    return eigenvalues

def observer_eigenvalues(A: np.ndarray, C: np.ndarray, L: np.ndarray):
    # Compute the eigenvalues of the observer error dynamics
    A_observer = A - L @ C
    eigenvalues = np.linalg.eigvals(A_observer)
    return eigenvalues

def check_closed_loop_stability(A: np.ndarray, B: np.ndarray, K: np.ndarray):
    eigenvalues = controller_eigenvalues(A, B, K)
    if np.all(np.real(eigenvalues) < 0):
        print("The closed-loop system is stable.")
    else:
        print("The closed-loop system is NOT stable.")
    return np.all(np.real(eigenvalues) < 0)

def vehicle_error(pos, pos_ref):
    """
    Compute the lateral error and heading error of the vehicle relative to a reference path.

    Parameters:
        pos (tuple): Current position of the vehicle (x, y, psi).
        pos_ref (tuple): Reference position on the path (x_ref, y_ref, psi_ref).
    Returns:
        e_y (float): Lateral error (distance from the reference path).
        e_psi (float): Heading error (difference in orientation).
    """
    x, y, psi = pos
    x_ref, y_ref, psi_ref = pos_ref

    dx = x - x_ref
    dy = y - y_ref

    e_y = -dx * np.sin(psi_ref) + dy * np.cos(psi_ref)

    e_psi_raw = psi - psi_ref
    e_psi = np.arctan2(np.sin(e_psi_raw), np.cos(e_psi_raw))

    return e_y, e_psi

def reconstruct_global_position(e_y, e_psi, x_ref, y_ref, psi_ref):
    """
    Reconstruct the global position of the vehicle from the lateral error and heading error.

    Parameters:
        e_y (float): Lateral error (distance from the reference path).
        e_psi (float): Heading error (difference in orientation).
        x_ref (float): Reference x position on the path.
        y_ref (float): Reference y position on the path.
        psi_ref (float): Reference heading angle on the path.
    Returns:
        x (float): Reconstructed global x position of the vehicle.
        y (float): Reconstructed global y position of the vehicle.
        psi (float): Reconstructed global heading angle of the vehicle.
    """
    # Reconstruct global position using the reference position and the errors
    x = x_ref - e_y * np.sin(psi_ref)
    y = y_ref + e_y * np.cos(psi_ref)
    psi = psi_ref + e_psi

    return x, y, psi

def main():
    A, B, E = Model.build_state_matrices()
    C, D = Model.build_output_matrices()

    Model.check_controllability(A, B)
    Model.check_open_loop_modes(A)
    Model.check_observability(A, C)

    # Build controller and observer gain matrices
    K = build_controller_matrix(A, B, controller_poles())
    L = build_observer_matrix(A, C, observer_poles())

    # Compute closed-loop eigenvalues
    eigenvalues = controller_eigenvalues(A, B, K)
    print("Closed-loop eigenvalues:", eigenvalues)

    # Compute observer eigenvalues
    observer_eigenvalues_list = observer_eigenvalues(A, C, L)
    print("Observer eigenvalues:", observer_eigenvalues_list)

    # Check closed-loop stability
    check_closed_loop_stability(A, B, K)

if __name__ == "__main__":
    main()