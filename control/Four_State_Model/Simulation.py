from . import Model
import control.Reference_States as Reference_States
import control.Vehicle_Params as Vehicle_Params
from . import Controller
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

def straight_line_path_test_case(dt=0.01, total_time=10.0):
    """
    Test case for a straight line path.
    """
    # Straight line path along the x-axis
    path_length = 100.0  # meters
    num_points = int(total_time / dt) + 1
    x_ref = np.linspace(0, path_length, num_points)
    y_ref = np.zeros_like(x_ref)  # y = 0 for straight line
    psi_ref = np.zeros_like(x_ref)  # heading along x-axis

    return x_ref, y_ref, psi_ref

def simulate_vehicle_dynamics(A, B, E, C, K, L, x0, x_hat0=None, dt=0.01, total_time=10.0):
    """
    Simulate the observer-based closed-loop dynamics of the linear vehicle error model.

    State vector:
        x = [e_y, e_psi, v_y, r]^T

    Control law:
        u = -K x_hat

    Plant:
        x_dot = A x + B u + E kappa

    Observer:
        x_hat_dot = A x_hat + B u + L(y - C x_hat)

    For this first simulation, the reference path is straight, so kappa = 0.

    Parameters:
        A, B, E:
            State-space matrices.
        C:
            Output matrix.
        K:
            State-feedback gain matrix.
        L:
            Observer gain matrix.
        x0:
            Initial true state.
        x_hat0:
            Initial observer estimate. If None, starts at zero.
        dt:
            Simulation timestep.
        total_time:
            Total simulation time.

    Returns:
        time:
            Time vector.
        x_history:
            True state history, shape (4, N).
        x_hat_history:
            Estimated state history, shape (4, N).
        y_history:
            Output history, shape (3, N).
        u_history:
            Steering input history, shape (N,).
        estimation_error_history:
            x - x_hat history, shape (4, N).
    """

    # Ensure consistent vector shapes
    B_vec = B.flatten()
    E_vec = E.flatten()

    x = np.asarray(x0, dtype=float).reshape(4).copy()

    if x_hat0 is None:
        x_hat = np.zeros(4)
    else:
        x_hat = np.asarray(x_hat0, dtype=float).reshape(4).copy()

    time = np.arange(0.0, total_time + dt, dt)
    num_steps = len(time)

    # History arrays
    x_history = np.zeros((4, num_steps))
    x_hat_history = np.zeros((4, num_steps))
    y_history = np.zeros((C.shape[0], num_steps))
    u_history = np.zeros(num_steps)
    estimation_error_history = np.zeros((4, num_steps))

    for i in range(num_steps):
        # Output from true plant
        y = C @ x

        # Control input from estimated state
        u = float((-K @ x_hat).item())

        # Store current values before integration
        x_history[:, i] = x
        x_hat_history[:, i] = x_hat
        y_history[:, i] = y
        u_history[i] = u
        estimation_error_history[:, i] = x - x_hat

        # Straight reference path for now
        kappa = 0.0

        # Plant dynamics
        x_dot = A @ x + B_vec * u + E_vec * kappa

        # Observer dynamics
        y_hat = C @ x_hat
        x_hat_dot = A @ x_hat + B_vec * u + L @ (y - y_hat)

        # Euler integration
        x = x + x_dot * dt
        x_hat = x_hat + x_hat_dot * dt

    return time, x_history, x_hat_history, y_history, u_history, estimation_error_history

def plot_simulation_position_straight_line(time, x_history, y_history, u_history):
    plt.figure(figsize=(12, 8))

    # Plot lateral error and heading error
    plt.subplot(3, 1, 1)
    plt.plot(time, x_history[0, :], label='Lateral Error $e_y$ (m)')
    plt.plot(time, np.degrees(x_history[1, :]), label='Heading Error $e_\\psi$ (deg)')
    plt.title('Vehicle Lateral and Heading Errors Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.grid()
    plt.legend()

    # Plot control input
    plt.subplot(3, 1, 2)
    plt.plot(time, np.degrees(u_history), label='Steering Input $\\delta$ (deg)')
    plt.title('Steering Input Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Angle (deg)')
    plt.grid()
    plt.legend()

    # Plot trajectory in XY plane
    plt.subplot(3, 1, 3)
    x_position = time * Vehicle_Params.V_x  # Assuming constant forward speed
    y_position = x_history[0, :]  # Lateral position is the lateral error
    plt.plot(x_position, y_position, label='Vehicle Trajectory')
    plt.title('Vehicle Trajectory in XY Plane')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():

    print("Running simulation for straight line path test case...")

    A, B, E = Model.build_state_matrices()
    C, D = Model.build_output_matrices()
    K = Controller.build_controller_matrix(A, B, Controller.controller_poles())
    L = Controller.build_observer_matrix(A, C, Controller.observer_poles())

    print("State matrix A:")
    print(A)

    print("\nInput matrix B:")
    print(B)

    print("\nDisturbance matrix E:")
    print(E)

    print("\nModel checks:")
    Model.check_controllability(A, B)
    Model.check_open_loop_modes(A)

    x0 = init_state()

    time, x_history, x_hat_history, y_history, u_history, estimation_error_history = simulate_vehicle_dynamics(
        A, B, E, C, K, L, x0, x_hat0=None, dt=0.01, total_time=10.0)

    plot_simulation_position_straight_line(time, x_history, y_history, u_history)

if __name__ == "__main__":
    main()