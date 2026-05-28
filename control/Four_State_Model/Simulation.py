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

def tyre_forces_with_friction_limit(x, delta, mu):
    """
    Compute front and rear lateral tyre forces with friction saturation.

    State:
        x = [e_y, e_psi, v_y, r]^T

    Input:
        delta = steering angle [rad]

    Returns:
        Fyf, Fyr:
            Saturated front and rear axle lateral forces.
        util_f, util_r:
            Front and rear tyre force utilisation ratios.
    """

    _, _, v_y, r = x

    Vx = Vehicle_Params.V_x
    m = Vehicle_Params.mass
    g = Vehicle_Params.g
    lf = Vehicle_Params.lf
    lr = Vehicle_Params.lr
    Cf = Vehicle_Params.Cf
    Cr = Vehicle_Params.Cr

    # Static axle normal loads
    wheelbase = lf + lr
    Fzf = m * g * lr / wheelbase
    Fzr = m * g * lf / wheelbase

    # Linear bicycle model slip-angle approximations
    alpha_f = delta - (v_y + lf * r) / Vx
    alpha_r = -(v_y - lr * r) / Vx

    # Linear tyre forces
    Fyf_linear = Cf * alpha_f
    Fyr_linear = Cr * alpha_r

    # Friction-limited maximum lateral forces
    Fyf_max = mu * Fzf
    Fyr_max = mu * Fzr

    # Utilisation before saturation
    util_f = abs(Fyf_linear) / Fyf_max if Fyf_max > 0 else np.inf
    util_r = abs(Fyr_linear) / Fyr_max if Fyr_max > 0 else np.inf

    # Saturated tyre forces
    Fyf = np.clip(Fyf_linear, -Fyf_max, Fyf_max)
    Fyr = np.clip(Fyr_linear, -Fyr_max, Fyr_max)

    return Fyf, Fyr, util_f, util_r

def nonlinear_friction_limited_dynamics(x, delta, mu, kappa=0.0):
    """
    Nonlinear/saturated dynamic bicycle error-state plant.
    """

    e_y, e_psi, v_y, r = x

    Vx = Vehicle_Params.V_x
    m = Vehicle_Params.mass
    Iz = Vehicle_Params.yaw_inertia
    lf = Vehicle_Params.lf
    lr = Vehicle_Params.lr

    Fyf, Fyr, util_f, util_r = tyre_forces_with_friction_limit(x, delta, mu)

    e_y_dot = v_y + Vx * e_psi
    e_psi_dot = r - Vx * kappa
    v_y_dot = (Fyf + Fyr) / m - Vx * r
    r_dot = (lf * Fyf - lr * Fyr) / Iz

    x_dot = np.array([e_y_dot, e_psi_dot, v_y_dot, r_dot])

    return x_dot, util_f, util_r


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

    # Import friction coefficient for nonlinear dynamics
    mu = Vehicle_Params.mu

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
    front_util_history = np.zeros(num_steps)
    rear_util_history = np.zeros(num_steps)
    traction_lost_history = np.zeros(num_steps, dtype=bool)

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
        #x_dot = A @ x + B_vec * u + E_vec * kappa
        x_dot, util_f, util_r = nonlinear_friction_limited_dynamics(x, u, mu, kappa)

        front_util_history[i] = util_f
        rear_util_history[i] = util_r
        traction_lost_history[i] = (util_f >= 1.0) or (util_r >= 1.0)


        # Observer dynamics
        y_hat = C @ x_hat
        x_hat_dot = A @ x_hat + B_vec * u + L @ (y - y_hat)

        # Euler integration
        x = x + x_dot * dt
        x_hat = x_hat + x_hat_dot * dt

    return time, x_history, x_hat_history, y_history, u_history, estimation_error_history, front_util_history, rear_util_history, traction_lost_history

def curvature_feedforward_steering(kappa):
    """
    Compute steady-state steering feedforward for the linear dynamic bicycle model.

    This gives the approximate steering angle required to follow a path
    with curvature kappa when lateral and heading errors are zero.
    """

    Vx = Vehicle_Params.V_x
    m = Vehicle_Params.mass
    lf = Vehicle_Params.lf
    lr = Vehicle_Params.lr
    Cf = Vehicle_Params.Cf
    Cr = Vehicle_Params.Cr

    wheelbase = lf + lr

    delta_ff = (
        wheelbase * kappa
        + (m * Vx**2 * kappa / wheelbase) * ((lr / Cf) - (lf / Cr))
    )

    return delta_ff

def simulate_path_tracking(
        A, B, E, C, K, L,
        x0, 
        path_ref, psi_ref, kappa_ref,
        x_hat0 = None, dt=0.01, total_time=10.0,
        use_feedforward=True,
        use_friction_limit=True
    ):
    """Simulate the observer-based closed-loop dynamics of the linear vehicle error model for a general reference path.
    The reference path is defined by the curvature profile kappa_ref, which can be used to compute a feedforward steering input.
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
        path_ref:
            Reference path positions (x_ref, y_ref).
        psi_ref:
            Reference path headings.
        kappa_ref:
            Reference path curvatures.
        x_hat0:
            Initial observer estimate. If None, starts at zero.
        dt:
            Simulation timestep.
        total_time:
            Total simulation time.
        use_feedforward:
            Whether to include curvature feedforward in the control input.
        use_friction_limit:
            Whether to simulate nonlinear dynamics with friction saturation.
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

    # Import friction coefficient for friction-limited dynamics
    mu = Vehicle_Params.mu

    # Unpack reference path
    x_ref_arr, y_ref_arr = path_ref
    psi_ref = psi_ref
    kappa_ref = kappa_ref

    x = np.asarray(x0, dtype=float).reshape(4).copy()

    if x_hat0 is None:
        x_hat = np.zeros(4)
    else: 
        x_hat = np.asarray(x_hat0, dtype=float).reshape(4).copy()

    # Time vector
    time = np.arange(0.0, total_time + dt, dt)
    num_steps = len(time)

    # History arrays
    x_history = np.zeros((4, num_steps))
    x_hat_history = np.zeros((4, num_steps))
    y_history = np.zeros((C.shape[0], num_steps))
    u_history = np.zeros(num_steps)
    delta_ff_history = np.zeros(num_steps)
    delta_fb_history = np.zeros(num_steps)
    estimation_error_history = np.zeros((4, num_steps))

    front_util_history = np.zeros(num_steps)
    rear_util_history = np.zeros(num_steps)
    traction_lost_history = np.zeros(num_steps, dtype=bool)

    # Global reconstruction histories for plotting
    x_global_history = np.zeros(num_steps)
    y_global_history = np.zeros(num_steps)
    psi_global_history = np.zeros(num_steps)

    x_ref_history = np.zeros(num_steps)
    y_ref_history = np.zeros(num_steps)
    psi_ref_history = np.zeros(num_steps)
    kappa_history = np.zeros(num_steps)

    for i in range(num_steps):
        # Protect against reference arrays shorter than the simulation
        idx = min(i, len(x_ref_arr) - 1, len(y_ref_arr) - 1, len(psi_ref) - 1, len(kappa_ref) - 1)

        x_ref_i = x_ref_arr[idx]
        y_ref_i = y_ref_arr[idx]
        psi_ref_i = psi_ref[idx]
        kappa_i = kappa_ref[idx]

        # Reconstruct approximate global pose from current error state
        pos = Controller.reconstruct_global_position(
            e_y=x[0],
            e_psi=x[1],
            x_ref=x_ref_i,
            y_ref=y_ref_i,
            psi_ref=psi_ref_i
        )

        pos_ref = (x_ref_i, y_ref_i, psi_ref_i)

        # Path-style measurement calculation
        e_y_meas, e_psi_meas = Controller.vehicle_error(pos, pos_ref)

        # Measured output: [e_y, e_psi, r]
        y = np.array([
            e_y_meas,
            e_psi_meas,
            x[3]
        ])

        # Estimated output
        y_hat = C @ x_hat

        # Feedback steering
        delta_fb = float((-K @ x_hat).item())

        # Feedforward steering for path curvature
        if use_feedforward:
            delta_ff = curvature_feedforward_steering(kappa_i)
        else:
            delta_ff = 0.0

        # Total steering command
        delta = delta_ff + delta_fb

        # Optional actuator saturation
        delta = np.clip(
            delta,
            -Vehicle_Params.max_steering_angle,
            Vehicle_Params.max_steering_angle
        )

        # Store current values before integration
        x_history[:, i] = x
        x_hat_history[:, i] = x_hat
        y_history[:, i] = y
        u_history[i] = delta
        delta_ff_history[i] = delta_ff
        delta_fb_history[i] = delta_fb
        estimation_error_history[:, i] = x - x_hat

        x_global_history[i] = pos[0]
        y_global_history[i] = pos[1]
        psi_global_history[i] = pos[2]

        x_ref_history[i] = x_ref_i
        y_ref_history[i] = y_ref_i
        psi_ref_history[i] = psi_ref_i
        kappa_history[i] = kappa_i

        # Plant dynamics
        if use_friction_limit:
            x_dot, util_f, util_r = nonlinear_friction_limited_dynamics(
                x, delta, mu, kappa_i
            )
        else:
            x_dot = A @ x + B_vec * delta + E_vec * kappa_i
            util_f = 0.0
            util_r = 0.0

        front_util_history[i] = util_f
        rear_util_history[i] = util_r
        traction_lost_history[i] = (util_f >= 1.0) or (util_r >= 1.0)

        # Observer dynamics
        # Important: observer also receives known path curvature input E*kappa
        x_hat_dot = (
            A @ x_hat
            + B_vec * delta
            + E_vec * kappa_i
            + L @ (y - y_hat)
        )

        # Euler integration
        x = x + x_dot * dt
        x_hat = x_hat + x_hat_dot * dt

        results = {
        "time": time,
        "x_history": x_history,
        "x_hat_history": x_hat_history,
        "y_history": y_history,
        "u_history": u_history,
        "delta_ff_history": delta_ff_history,
        "delta_fb_history": delta_fb_history,
        "estimation_error_history": estimation_error_history,
        "front_util_history": front_util_history,
        "rear_util_history": rear_util_history,
        "traction_lost_history": traction_lost_history,
        "x_global_history": x_global_history,
        "y_global_history": y_global_history,
        "psi_global_history": psi_global_history,
        "x_ref_history": x_ref_history,
        "y_ref_history": y_ref_history,
        "psi_ref_history": psi_ref_history,
        "kappa_history": kappa_history,
    }
        
    return results

def plot_path_tracking(results):
    time = results["time"]
    x_ref_history = results["x_ref_history"]
    y_ref_history = results["y_ref_history"]
    x_global_history = results["x_global_history"]
    y_global_history = results["y_global_history"]

    plt.plot(x_global_history, y_global_history, label='Vehicle Trajectory')
    plt.plot(x_ref_history, y_ref_history, "k--", label="Reference path")
    plt.title('Vehicle Trajectory in XY Plane')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid()
    plt.legend()

    plt.show()

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
    plt.plot(x_position, np.zeros_like(x_position), "k--", label="Reference path")
    plt.title('Vehicle Trajectory in XY Plane')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_tire_utilization(time, front_util_history, rear_util_history):
    plt.figure(figsize=(10, 4))
    plt.plot(time, front_util_history, label="Front tyre utilisation")
    plt.plot(time, rear_util_history, label="Rear tyre utilisation")
    plt.axhline(1.0, color="k", linestyle="--", label="Traction limit")
    plt.xlabel("Time (s)")
    plt.ylabel("Force utilisation")
    plt.title("Tyre Force Utilisation")
    plt.grid()
    plt.legend()
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
    Model.check_observability(A, C)

    x0 = init_state()
    '''
    time, x_history, x_hat_history, y_history, u_history, estimation_error_history, front_util_history, rear_util_history, traction_lost_history = simulate_vehicle_dynamics(
        A, B, E, C, K, L, x0, x_hat0=None, dt=0.01, total_time=10.0)
    '''

    #plot_simulation_position_straight_line(time, x_history, y_history, u_history)
    #plot_tire_utilization(time, front_util_history, rear_util_history)

    u, path, curvature, radius, heading, length, time = Reference_States.init_path()

    print(f"Simulating path tracking for reference path of length {length:.2f} m and estimated traversal time {time:.2f} s...")

    results = simulate_path_tracking(
        A, B, E, C, K, L,
        x0, 
        path_ref=(path[:, 0], path[:, 1]),
        psi_ref=heading,
        kappa_ref=curvature,
        x_hat0=None, dt=0.01, total_time=time,
        use_feedforward=True,
        use_friction_limit=True
    )

    plot_path_tracking(results)

if __name__ == "__main__":
    main()