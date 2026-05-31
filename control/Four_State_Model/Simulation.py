from . import Model
import control.Reference_States as Reference_States
import control.Vehicle_Params as Vehicle_Params
from . import Controller
import numpy as np
import matplotlib.pyplot as plt
from . import plotting


def init_state():
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
    return Vehicle_Params.initial_state

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

def path_arclength(x_ref_arr, y_ref_arr):
    path = np.column_stack((x_ref_arr, y_ref_arr))
    ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(ds)))


def curvature_steady_state(A, B, E, kappa):
    """
    Compute the steady-state error-state vector and steering angle
    for a path curvature kappa.

    Solves:
        A x_ss + B delta_ff + E kappa = 0

    with:
        e_y_ss = 0

    Unknowns:
        [e_y_ss, e_psi_ss, v_y_ss, r_ss, delta_ff]
    """
    B_col = B.reshape(4, 1)
    E_vec = E.reshape(4)

    M = np.block([
        [A, B_col],
        [np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])]
    ])

    rhs = np.concatenate((-E_vec * kappa, [0.0]))

    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(M, rhs, rcond=None)[0]

    x_ss = sol[:4]
    delta_ff = sol[4]

    return x_ss, delta_ff

def make_time_vector(total_time, dt):
    """
    Build a time vector that does not overshoot total_time by an extra dt.
    """
    if total_time <= 0:
        return np.array([0.0])

    n_steps = int(np.floor(total_time / dt)) + 1
    time_vec = np.arange(n_steps) * dt

    # Append exact final time only if it is not already represented.
    if time_vec[-1] < total_time - 1e-12:
        time_vec = np.append(time_vec, total_time)

    return time_vec


def choose_valid_tracking_distance(
        s_ref,
        kappa_ref,
        Vx,
        mu,
        g,
        endpoint_buffer=2.0,
        utilisation_margin=0.90
    ):
    """
    Select the final arclength to use for simulation.

    The simulation is stopped before:
        1. the final endpoint, where spline/finite-difference curvature
           can be unreliable; and
        2. any terminal region whose curvature requires more lateral
           acceleration than the tyre friction margin allows.

    Curvature feasibility is approximated using:
        ay = Vx^2 * kappa <= utilisation_margin * mu * g
    """

    s_ref = np.asarray(s_ref, dtype=float)
    kappa_ref = np.asarray(kappa_ref, dtype=float)

    s_total = s_ref[-1]

    # Feasible curvature limit based on lateral acceleration demand.
    kappa_limit = utilisation_margin * mu * g / (Vx ** 2)

    # Default: stop before the endpoint.
    s_stop = max(0.0, s_total - endpoint_buffer)

    # If the terminal part of the path exceeds the curvature limit,
    # stop before the start of that terminal infeasible region.
    terminal_bad = np.abs(kappa_ref) > kappa_limit

    if terminal_bad[-1]:
        # Walk backwards until curvature becomes feasible again.
        first_bad_idx = len(kappa_ref) - 1

        while first_bad_idx > 0 and terminal_bad[first_bad_idx - 1]:
            first_bad_idx -= 1

        # Stop at the last feasible point before the terminal bad region.
        stop_idx = max(0, first_bad_idx - 1)
        s_stop = min(s_stop, s_ref[stop_idx])

    s_stop = max(0.0, s_stop)

    return s_stop, kappa_limit

def simulate_path_tracking(
        A, B, E, C, K, L,
        x0,
        path_ref, psi_ref, kappa_ref,
        x_hat0=None,
        dt=0.01,
        total_time=10.0,
        use_feedforward=True,
        use_friction_limit=False,
        endpoint_buffer=2.0,
        utilisation_margin=0.90
    ):

    B_vec = B.flatten()
    E_vec = E.flatten()

    mu = Vehicle_Params.mu
    g = Vehicle_Params.g
    Vx = Vehicle_Params.V_x

    x_ref_arr, y_ref_arr = path_ref
    x_ref_arr = np.asarray(x_ref_arr, dtype=float)
    y_ref_arr = np.asarray(y_ref_arr, dtype=float)
    psi_ref = np.unwrap(np.asarray(psi_ref, dtype=float))
    kappa_ref = np.asarray(kappa_ref, dtype=float)

    if not (
        len(x_ref_arr) == len(y_ref_arr) ==
        len(psi_ref) == len(kappa_ref)
    ):
        raise ValueError(
            "Reference arrays must have matching lengths: "
            f"x={len(x_ref_arr)}, y={len(y_ref_arr)}, "
            f"psi={len(psi_ref)}, kappa={len(kappa_ref)}"
        )

    s_ref = path_arclength(x_ref_arr, y_ref_arr)
    s_total = s_ref[-1]

    # Stop before terminal endpoint saturation or terminal infeasible curvature.
    s_stop, kappa_limit = choose_valid_tracking_distance(
        s_ref=s_ref,
        kappa_ref=kappa_ref,
        Vx=Vx,
        mu=mu,
        g=g,
        endpoint_buffer=endpoint_buffer,
        utilisation_margin=utilisation_margin
    )

    # Do not run beyond the valid reference distance.
    valid_total_time = s_stop / Vx
    total_time = min(total_time, valid_total_time)

    time_vec = make_time_vector(total_time, dt)
    num_steps = len(time_vec)

    x = np.asarray(x0, dtype=float).reshape(4).copy()

    if x_hat0 is None:
        x_hat = np.zeros(4)
    else:
        x_hat = np.asarray(x_hat0, dtype=float).reshape(4).copy()

    x_history = np.zeros((4, num_steps))
    x_hat_history = np.zeros((4, num_steps))
    y_history = np.zeros((C.shape[0], num_steps))
    u_history = np.zeros(num_steps)
    delta_ff_history = np.zeros(num_steps)
    delta_fb_history = np.zeros(num_steps)
    estimation_error_history = np.zeros((4, num_steps))

    a_max_history = np.zeros(num_steps)

    front_util_history = np.zeros(num_steps)
    rear_util_history = np.zeros(num_steps)
    traction_lost_history = np.zeros(num_steps, dtype=bool)

    x_global_history = np.zeros(num_steps)
    y_global_history = np.zeros(num_steps)
    psi_global_history = np.zeros(num_steps)

    x_ref_history = np.zeros(num_steps)
    y_ref_history = np.zeros(num_steps)
    psi_ref_history = np.zeros(num_steps)
    kappa_history = np.zeros(num_steps)
    s_history = np.zeros(num_steps)

    for i, t in enumerate(time_vec):

        # Advance along the valid portion of the path only.
        s_i = Vx * t
        s_i = min(s_i, s_stop)
        s_history[i] = s_i

        x_ref_i = np.interp(s_i, s_ref, x_ref_arr)
        y_ref_i = np.interp(s_i, s_ref, y_ref_arr)
        psi_ref_i = np.interp(s_i, s_ref, psi_ref)
        kappa_i = np.interp(s_i, s_ref, kappa_ref)

        if use_feedforward:
            x_ss, delta_ff = curvature_steady_state(A, B, E, kappa_i)
        else:
            x_ss = np.zeros(4)
            delta_ff = 0.0

        # Measurement model.
        y = C @ x
        y_hat = C @ x_hat

        # Feedback on deviation from curved-path steady state.
        x_tilde_hat = x_hat - x_ss
        delta_fb = float((-K @ x_tilde_hat).item())

        delta_unsat = delta_ff + delta_fb
        delta = np.clip(
            delta_unsat,
            -Vehicle_Params.max_steering_angle,
            Vehicle_Params.max_steering_angle
        )

        pos = Controller.reconstruct_global_position(
            e_y=x[0],
            e_psi=x[1],
            x_ref=x_ref_i,
            y_ref=y_ref_i,
            psi_ref=psi_ref_i
        )

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

        if use_friction_limit:
            x_dot, util_f, util_r = nonlinear_friction_limited_dynamics(
                x, delta, mu, kappa_i
            )
        else:
            x_dot = A @ x + B_vec * delta + E_vec * kappa_i
            util_f = 0.0
            util_r = 0.0
        a_max_history[i] = np.linalg.norm(x_dot[2:])  # Lateral acceleration magnitude
        front_util_history[i] = util_f
        rear_util_history[i] = util_r
        traction_lost_history[i] = (util_f >= 1.0) or (util_r >= 1.0)

        x_hat_dot = (
            A @ x_hat
            + B_vec * delta
            + E_vec * kappa_i
            + L @ (y - y_hat)
        )

        x = x + x_dot * dt
        x_hat = x_hat + x_hat_dot * dt

    results = {
        "time": time_vec,
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
        "s_history": s_history,
        "x_ref_full": x_ref_arr,
        "y_ref_full": y_ref_arr,
        "psi_ref_full": psi_ref,
        "kappa_ref_full": kappa_ref,
        "s_ref": s_ref,
        "s_total": s_total,
        "s_stop": s_stop,
        "kappa_feasibility_limit": kappa_limit,
        "endpoint_buffer": endpoint_buffer,
        "utilisation_margin": utilisation_margin,
        "a_max_history": a_max_history,
    }

    return results


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

    print("\nOutput-Input matrix D:")
    print(D)

    print("\n Feedforward matrix E:")
    print(E)

    print("\nOutput matrix C:")
    print(C)

    print("\nState-feedback gain K:")
    print(K)

    print("\nObserver gain L:")
    print(L)

    print("\nController poles:")
    print(Controller.controller_poles())

    print("\nObserver poles:")
    print(Controller.observer_poles())

    print("\nClosed-loop (Controller) eigenvalues:")
    print(Controller.controller_eigenvalues(A, B, K))

    print("\nObserver eigenvalues:")
    print(Controller.observer_eigenvalues(A, C, L))


    print("\nModel checks:")
    Model.check_controllability(A, B)
    Model.check_open_loop_modes(A)
    Model.check_observability(A, C)

    A2, B2, E2 = Model.build_state_matrices("0.8")
    A3, B3, E3 = Model.build_state_matrices("1.2")

    u, path, curvature, radius, heading, length, time = Reference_States.init_path() #Mellow path
    u_m, path_m, curvature_m, radius_m, heading_m, length_m, time_m = Reference_States.init_path_mellow() #This is actually the aggressive path

    # Start the vehicle and observer at the steady-state condition
    # corresponding to the initial path curvature. This removes the
    # artificial startup transient caused by starting x_hat at zero.
    kappa0 = curvature[0]
    x_ss0, delta_ff0 = curvature_steady_state(A, B, E, kappa0)

    x0 = x_ss0.copy()
    #x_hat0 = x_ss0.copy()
    #x_hat0 = np.zeros_like(x0)
    x_hat0 = x0 + np.array([0.2, np.radians(5), 0.0, 0.0])

    print(f"Initial curvature: {kappa0:.6f} 1/m")
    print(f"Initial steady-state x0: {x0}")
    print(f"Initial feedforward steering: {np.degrees(delta_ff0):.3f} deg")

    print(f"Simulating path tracking for reference path of length {length:.2f} m and estimated traversal time {time:.2f} s...")

    results = simulate_path_tracking(
        A, B, E, C, K, L,
        x0,
        path_ref=(path[:, 0], path[:, 1]),
        psi_ref=heading,
        kappa_ref=curvature,
        x_hat0=x_hat0,
        dt=0.01,
        total_time=time,
        use_feedforward=True,
        use_friction_limit=True,
        endpoint_buffer=8.0,
        utilisation_margin=0.90
    )
    results2 = simulate_path_tracking(
        A2, B2, E2, C, K, L,
        x0,
        path_ref=(path[:, 0], path[:, 1]),
        psi_ref=heading,
        kappa_ref=curvature,
        x_hat0=x_hat0,
        dt=0.01,
        total_time=time,
        use_feedforward=True,
        use_friction_limit=True,
        endpoint_buffer=8.0,
        utilisation_margin=0.90
    )
    results3 = simulate_path_tracking(
        A3, B3, E3, C, K, L,
        x0,
        path_ref=(path[:, 0], path[:, 1]),
        psi_ref=heading,
        kappa_ref=curvature,
        x_hat0=x_hat0,
        dt=0.01,
        total_time=time,
        use_feedforward=True,
        use_friction_limit=True,
        endpoint_buffer=8.0,
        utilisation_margin=0.90
    )

    results4 = simulate_path_tracking(
        A, B, E, C, K, L,
        x0,
        path_ref=(path[:, 0], path[:, 1]),
        psi_ref=heading,
        kappa_ref=curvature,
        x_hat0=x_hat0,
        dt=0.01,
        total_time=time,
        use_feedforward=True,
        use_friction_limit=False,
        endpoint_buffer=8.0,
        utilisation_margin=0.90
    )

    results5 = simulate_path_tracking(
        A, B, E, C, K, L,
        x0,
        path_ref=(path_m[:, 0], path_m[:, 1]),
        psi_ref=heading_m,
        kappa_ref=curvature_m,
        x_hat0=x_hat0,
        dt=0.01,
        total_time=time_m,   # <-- use mellow path time here
        use_feedforward=True,
        use_friction_limit=True,
        endpoint_buffer=8.0,
        utilisation_margin=0.90
    )

    results6 = simulate_path_tracking(
        A, B, E, C, K, L,
        x0,
        path_ref=(path_m[:, 0], path_m[:, 1]),
        psi_ref=heading_m,
        kappa_ref=curvature_m,
        x_hat0=x_hat0,
        dt=0.01,
        total_time=time_m,   # <-- use mellow path time here
        use_feedforward=True,
        use_friction_limit=False,
        endpoint_buffer=8.0,
        utilisation_margin=0.90
    )


    print(f"Full path length: {results['s_total']:.2f} m")
    print(f"Simulated valid path length: {results['s_stop']:.2f} m")
    print(f"Curvature feasibility limit: {results['kappa_feasibility_limit']:.5f} 1/m")

    plotting.observer_validation(results)
    
    """
    print("\nnominal perf (mellow path no sat) \n")
    plotting.print_performance(results4)

    print("\nnominal perf (mellow path sat) \n")
    plotting.print_performance(results)

    print("\naggressive path sat \n")
    plotting.print_performance(results5)

    print("\naggressive path no sat \n")
    plotting.print_performance(results6)

    print("\nlow tyre stiffness perf \n")
    plotting.print_performance(results2)

    print("\nhigh tyre stiffness perf \n")
    plotting.print_performance(results3)
    #plotting.plot_variance_in_C(results, results2, results3)
    

    plotting.plot_path_tracking(results)
    plotting.plot_path_tracking_diagnostics(results)
    plotting.plot_tire_utilization(
    results["time"],
    results["front_util_history"],
    results["rear_util_history"]
    )
    plotting.plot_acceleration_history(
        results["time"],
        results["a_max_history"]
    )
    

    plotting.nominal_model_comparison_plots(
        results=results,
        results_nominal=results_nominal,
        ey_limit=0.3,
        epsi_limit_deg=5,
        steering_limit_deg=36,
        save_path="fig1_nominal_model_comparison.pdf"
    )
    """
    '''
    plotting.friction_comparison_plots(
        time=results["time"],
        time_other=results_nominal["time"],
        front_util_history=results["front_util_history"],
        front_util_other=results_nominal["front_util_history"],
        rear_util_history=results["rear_util_history"],
        rear_util_other=results_nominal["rear_util_history"],
        a_max_history=results["a_max_history"],
        a_max_other=results_nominal["a_max_history"],
    )
    '''
    
if __name__ == "__main__":
    main()