from . import Model
import control.Reference_States as Reference_States
import control.Vehicle_Params as Vehicle_Params
import numpy as np
import matplotlib.pyplot as plt


def init_state(
    e_y0=0.8,
    e_psi0=np.radians(3.0),
    v_y0=0.0,
    r0=0.0
):
    """
    Initial state for the 4-state dynamic bicycle error model.

    State:
        x = [e_y, e_psi, v_y, r]^T

    where:
        e_y   = lateral error [m]
        e_psi = heading error [rad]
        v_y   = lateral velocity at CG [m/s]
        r     = yaw rate [rad/s]
    """

    return np.array([
        [e_y0],
        [e_psi0],
        [v_y0],
        [r0]
    ], dtype=float)


def main():

    A, B, E = Model.build_state_matrices()

    print("State matrix A:")
    print(A)

    print("\nInput matrix B:")
    print(B)

    print("\nDisturbance matrix E:")
    print(E)

    print("\nModel checks:")
    Model.check_controllability(A, B)
    Model.check_open_loop_modes(A)


if __name__ == "__main__":
    main()