import matplotlib as plt
import numpy as np
import control.Vehicle_Params as Vehicle_Params

def plot_path_tracking(results):
    plt.figure(figsize=(10, 4.5))

    plt.plot(
        results["x_ref_full"],
        results["y_ref_full"],
        "k--",
        linewidth=1.5,
        label="Full reference path"
    )

    plt.plot(
        results["x_ref_history"],
        results["y_ref_history"],
        linestyle=":",
        linewidth=1.2,
        label="Time-marched reference point"
    )

    plt.plot(
        results["x_global_history"],
        results["y_global_history"],
        linewidth=1.8,
        label="Vehicle trajectory"
    )

    plt.title("Vehicle Trajectory in XY Plane")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_path_tracking_diagnostics(results):
    time = results["time"]
    x_history = results["x_history"]

    plt.figure(figsize=(10, 7))

    plt.subplot(3, 1, 1)
    plt.plot(time, x_history[0, :], label="e_y")
    plt.ylabel("Lateral error [m]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time, np.degrees(x_history[1, :]), label="e_psi")
    plt.ylabel("Heading error [deg]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time, np.degrees(results["u_history"]), label="delta")
    plt.plot(time, np.degrees(results["delta_ff_history"]), "--", label="delta_ff")
    plt.plot(time, np.degrees(results["delta_fb_history"]), ":", label="delta_fb")
    plt.ylabel("Steering [deg]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
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

def plot_acceleration_history(time, a_max_history):
    plt.figure(figsize=(10, 4))
    plt.plot(time, a_max_history, label="Lateral acceleration magnitude")
    plt.axhline(Vehicle_Params.mu * Vehicle_Params.g, color="k", linestyle="--", label="Friction limit")
    plt.xlabel("Time (s)")
    plt.ylabel("Lateral acceleration (m/s^2)")
    plt.title("Lateral Acceleration History")
    plt.grid()
    plt.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def nominal_model_comparison_plots(
    results,
    results_nominal,
    ey_limit=None,
    epsi_limit_deg=None,
    steering_limit_deg=36,
    lane_y_min=1,
    lane_y_max=7,
    save_path=None
):
    """
    Fig. 1: Nominal model comparison.

    Assumptions:
        results          = nonlinear/original model result dictionary
        results_nominal  = linear/simplified model result dictionary

        x_history[0, :] = lateral error e_y [m]
        x_history[1, :] = heading error e_psi [rad]

    Optional:
        ey_limit: lateral error limit in m. If None, no e_y limit lines are drawn.
        epsi_limit_deg: heading error limit in deg. If None, no e_psi limit lines are drawn.
        steering_limit_deg: steering limit in deg.
    """

    # -----------------------------
    # Helper functions
    # -----------------------------
    def get_first_available(data, keys):
        for key in keys:
            if key in data:
                return np.asarray(data[key])
        raise KeyError(f"None of the following keys were found: {keys}")

    def as_1d(arr):
        arr = np.asarray(arr)
        return np.squeeze(arr)

    def steering_to_deg(delta):
        delta = as_1d(delta)

        # If steering magnitude looks like radians, convert to degrees.
        # If it is already in degrees, leave it unchanged.
        if np.nanmax(np.abs(delta)) < 2 * np.pi:
            return np.degrees(delta)
        return delta

    # -----------------------------
    # Extract nonlinear/original data
    # -----------------------------
    time_nl = as_1d(results["time"])
    x_history_nl = np.asarray(results["x_history"])

    x_global_nl = as_1d(results["x_global_history"])
    y_global_nl = as_1d(results["y_global_history"])

    ey_nl = as_1d(x_history_nl[0, :])
    epsi_nl_deg = np.degrees(as_1d(x_history_nl[1, :]))

    delta_nl = get_first_available(
        results,
        ["delta_history", "delta", "u_history", "steering_history"]
    )
    delta_nl_deg = steering_to_deg(delta_nl)

    # -----------------------------
    # Extract linear/simplified data
    # -----------------------------
    time_lin = as_1d(results_nominal["time"])
    x_history_lin = np.asarray(results_nominal["x_history"])

    x_global_lin = as_1d(results_nominal["x_global_history"])
    y_global_lin = as_1d(results_nominal["y_global_history"])

    ey_lin = as_1d(x_history_lin[0, :])
    epsi_lin_deg = np.degrees(as_1d(x_history_lin[1, :]))

    delta_lin = get_first_available(
        results_nominal,
        ["delta_history", "delta", "u_history", "steering_history"]
    )
    delta_lin_deg = steering_to_deg(delta_lin)

    # -----------------------------
    # Reference path
    # -----------------------------
    x_ref = as_1d(results["x_ref_full"])
    y_ref = as_1d(results["y_ref_full"])

    # -----------------------------
    # Plot styling
    # -----------------------------
    colour_nl = "tab:blue"      # nonlinear/original
    colour_lin = "tab:orange"   # linear/simplified

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    # ==========================================================
    # 1) XY trajectory
    # ==========================================================
    ax = axes[0]

    ax.plot(
        x_ref,
        y_ref,
        "k--",
        linewidth=1.4,
        label="Reference path"
    )

    ax.plot(
        x_global_lin,
        y_global_lin,
        color=colour_lin,
        linewidth=1.8,
        label="Linear/simplified model"
    )

    ax.plot(
        x_global_nl,
        y_global_nl,
        color=colour_nl,
        linewidth=1.8,
        label="Nonlinear/original model"
    )

    ax.axhline(
        lane_y_min,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Lane boundaries"
    )

    ax.axhline(
        lane_y_max,
        color="black",
        linestyle="--",
        linewidth=1.0
    )

    ax.set_title("Nominal Model Comparison: Vehicle Trajectory")
    ax.set_xlabel("X position [m]")
    ax.set_ylabel("Y position [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_ylim(0, 8)
    ax.set_xlim(0, 90)

    # ==========================================================
    # 2) e_y and e_psi on dual axes
    # ==========================================================
    ax1 = axes[1]
    ax2 = ax1.twinx()

    # Lateral error: solid lines
    l1, = ax1.plot(
        time_lin,
        ey_lin,
        color=colour_lin,
        linestyle="-",
        linewidth=1.8,
        label=r"Linear $e_y$"
    )

    l2, = ax1.plot(
        time_nl,
        ey_nl,
        color=colour_nl,
        linestyle="-",
        linewidth=1.8,
        label=r"Nonlinear $e_y$"
    )

    # Heading error: dashed lines
    l3, = ax2.plot(
        time_lin,
        epsi_lin_deg,
        color=colour_lin,
        linestyle="--",
        linewidth=1.8,
        label=r"Linear $e_\psi$"
    )

    l4, = ax2.plot(
        time_nl,
        epsi_nl_deg,
        color=colour_nl,
        linestyle="--",
        linewidth=1.8,
        label=r"Nonlinear $e_\psi$"
    )

    # Optional design-limit lines
    if ey_limit is not None:
        ax1.axhline(
            ey_limit,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=r"$e_y$ limit"
        )
        ax1.axhline(
            -ey_limit,
            color="black",
            linestyle="--",
            linewidth=1.0
        )

    if epsi_limit_deg is not None:
        ax2.axhline(
            epsi_limit_deg,
            color="black",
            linestyle=":",
            linewidth=1.0,
            label=r"$e_\psi$ limit"
        )
        ax2.axhline(
            -epsi_limit_deg,
            color="black",
            linestyle=":",
            linewidth=1.0
        )

    ax1.set_title(r"Tracking Errors: $e_y$ and $e_\psi$")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(r"Lateral error $e_y$ [m]")
    ax1.set_ylim(-1, 1)
    ax2.set_ylabel(r"Heading error $e_\psi$ [deg]")
    ax2.set_ylim(-30, 30)

    ax1.grid(True)

    # Combined legend from both axes
    lines = [l1, l2, l3, l4]
    labels = [line.get_label() for line in lines]

    if ey_limit is not None:
        lines.append(ax1.lines[-2])
        labels.append(r"$e_y$ limit")

    if epsi_limit_deg is not None:
        lines.append(ax2.lines[-2])
        labels.append(r"$e_\psi$ limit")

    ax1.legend(lines, labels, loc="best")

    # ==========================================================
    # 3) Steering input
    # ==========================================================
    ax = axes[2]

    ax.plot(
        time_lin,
        delta_lin_deg,
        color=colour_lin,
        linewidth=1.8,
        label="Linear/simplified steering"
    )

    ax.plot(
        time_nl,
        delta_nl_deg,
        color=colour_nl,
        linewidth=1.8,
        label="Nonlinear/original steering"
    )

    ax.axhline(
        steering_limit_deg,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=rf"Steering limit $\pm {steering_limit_deg:.0f}^\circ$"
    )

    ax.axhline(
        -steering_limit_deg,
        color="black",
        linestyle="--",
        linewidth=1.0
    )

    ax.set_title("Controller Steering Input")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Steering input $\delta$ [deg]")
    ax.grid(True)
    ax.legend(loc="best")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def friction_comparison_plots(
    time,
    time_other,
    front_util_history,
    front_util_other,
    rear_util_history,
    rear_util_other,
    a_max_history,
    a_max_other,
):
    time = np.squeeze(time)
    time_other = np.squeeze(time_other)

    front_util_history = np.squeeze(front_util_history)
    front_util_other = np.squeeze(front_util_other)
    rear_util_history = np.squeeze(rear_util_history)
    rear_util_other = np.squeeze(rear_util_other)
    a_max_history = np.squeeze(a_max_history)
    a_max_other = np.squeeze(a_max_other)

    plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    plt.plot(
        time,
        front_util_history,
        label="Aggressive path - Front tyre utilisation",
        color="tab:blue"
    )
    plt.plot(
        time,
        rear_util_history,
        label="Aggressive path - Rear tyre utilisation",
        color="tab:blue",
        linestyle="--"
    )
    plt.plot(
        time_other,
        front_util_other,
        label="Mellow path - Front tyre utilisation",
        color="tab:orange"
    )
    plt.plot(
        time_other,
        rear_util_other,
        label="Mellow path - Rear tyre utilisation",
        color="tab:orange",
        linestyle="--"
    )
    plt.axhline(1.0, color="k", linestyle="--", linewidth=1.0, label="Friction limit")
    plt.ylabel("Force utilisation")
    plt.title("Tyre Force Utilisation Comparison")
    plt.ylim(0, 4)
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(
        time,
        a_max_history,
        label="Aggressive path",
        color="tab:blue"
    )
    plt.plot(
        time_other,
        a_max_other,
        label="Mellow path",
        color="tab:orange"
    )
    plt.axhline(
        Vehicle_Params.mu * Vehicle_Params.g,
        color="k",
        linestyle="--",
        linewidth=1.0,
        label=r"Friction limit $\mu g$"
    )
    plt.xlabel("Time [s]")
    plt.ylabel(r"Lateral acceleration [m/s$^2$]")
    plt.title("Lateral Acceleration History")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_variance_in_C(results, results2, results3):
    time = results["time"]
    x_history = results["x_history"]
    time2 = results2["time"]
    x_history2 = results2["x_history"]
    time3 = results3["time"]
    x_history3 = results3["x_history"]

    plt.figure(figsize=(10, 7))

    plt.subplot(3, 1, 1)
    plt.plot(
        results["x_ref_full"],
        results["y_ref_full"],
        "k--",
        linewidth=1.5,
        label="Full reference path"
    )
    plt.plot(
        results["x_global_history"],
        results["y_global_history"],
        linewidth=1.8,
        label="[Cf, Cr] = [Cf, Cr]"
    )
    plt.plot(
        results2["x_global_history"],
        results2["y_global_history"],
        linewidth=1.8,
        label="[Cf, Cr] = 0.8*[Cf, Cr]"
    )
    plt.plot(
        results3["x_global_history"],
        results3["y_global_history"],
        linewidth=1.8,
        label="[Cf, Cr] = 1.2*[Cf, Cr]"
    )
    plt.title("Vehicle Trajectory in XY Plane")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time, x_history[0, :], label="[Cf, Cr] = [Cf, Cr]")
    plt.plot(time2, x_history2[0, :], label="[Cf, Cr] = 0.8*[Cf, Cr]")
    plt.plot(time3, x_history3[0, :], label="[Cf, Cr] = 1.2*[Cf, Cr]")
    plt.ylabel("Lateral error (e_y) [m]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time, np.degrees(results["u_history"]), label="[Cf, Cr] = [Cf, Cr]")
    plt.plot(time2, np.degrees(results2["u_history"]), label="[Cf, Cr] = 0.8*[Cf, Cr]")
    plt.plot(time3, np.degrees(results3["u_history"]), label="[Cf, Cr] = 1.2*[Cf, Cr]")

    plt.ylabel("Steering (delta) [deg]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def print_performance(results):
    e_y = results["x_history"][0, :]
    e_psi = results["x_history"][1, :]
    delta = results["u_history"]
    a_max = results["a_max_history"]
    front_util_history=results["front_util_history"]
    rear_util_history=results["rear_util_history"]

    print('Maximum lateral error [m]:', max(abs(e_y)))
    print('Lateral error RMS [m]: ', np.sqrt(np.mean(np.square(e_y))))
    print('Maximum heading error [deg]: ', np.rad2deg(max(abs(e_psi))))
    print('Maximum steering input [deg]: ', np.rad2deg(max(abs(delta))))
    print('Maximum lateral acceleration [m/s^2]: ', max(abs(a_max)))
    print("Maximum front tyre utilisation:", max(front_util_history))
    print("Maximum rear tyre utilisation:", max(rear_util_history))

import numpy as np
import matplotlib.pyplot as plt


def observer_validation(results, save_path=None):
    time = np.squeeze(results["time"])

    x_history = np.asarray(results["x_history"])
    x_hat_history = np.asarray(results["x_hat_history"])

    # Assumed state ordering:
    # x[0, :] = e_y [m]
    # x[1, :] = e_psi [rad]
    e_y = np.squeeze(x_history[0, :])
    e_psi = np.squeeze(x_history[1, :])

    e_yhat = np.squeeze(x_hat_history[0, :])
    e_psihat = np.squeeze(x_hat_history[1, :])

    e_y_diff = e_y - e_yhat
    e_psi_diff_deg = np.degrees(e_psi - e_psihat)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Observer Estimation Error Convergence")

    axes[0].plot(
        time,
        e_y_diff,
        linewidth=1.8,
        label=r"$\tilde{e}_y = e_y - \hat{e}_y$"
    )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel(r"Lateral estimation error [m]")
    axes[0].grid(True)
    axes[0].legend(loc="best")

    axes[1].plot(
        time,
        e_psi_diff_deg,
        linewidth=1.8,
        label=r"$\tilde{e}_\psi = e_\psi - \hat{e}_\psi$"
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel(r"Heading estimation error [deg]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True)
    axes[1].legend(loc="best")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_friction_step_disturbance_comparison(
        results_no_sat,
        results_sat,
        ey_limit=0.30,
        steering_limit_deg=36,
        save_path=None
    ):
    """
    Compare the same friction-step disturbance with tyre saturation disabled
    and enabled.

    results_no_sat:
        Simplified/linear-style case where friction saturation is disabled.

    results_sat:
        Original/nonlinear-style case where tyre saturation is enabled.

    The plot is designed for the Numerical Validation section:
        1. path tracking trajectory,
        2. lateral error,
        3. steering input,
        4. tyre utilisation and friction step.
    """

    def as_1d(arr):
        return np.squeeze(np.asarray(arr))

    time_no = as_1d(results_no_sat["time"])
    time_sat = as_1d(results_sat["time"])

    ey_no = as_1d(results_no_sat["x_history"][0, :])
    ey_sat = as_1d(results_sat["x_history"][0, :])

    delta_no_deg = np.degrees(as_1d(results_no_sat["u_history"]))
    delta_sat_deg = np.degrees(as_1d(results_sat["u_history"]))

    front_no = as_1d(results_no_sat["front_util_history"])
    rear_no = as_1d(results_no_sat["rear_util_history"])
    front_sat = as_1d(results_sat["front_util_history"])
    rear_sat = as_1d(results_sat["rear_util_history"])

    mu_sat = as_1d(results_sat["mu_history"])

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), constrained_layout=True)

    # ------------------------------------------------------------
    # 1. XY path tracking
    # ------------------------------------------------------------
    ax = axes[0]

    ax.plot(
        results_sat["x_ref_full"],
        results_sat["y_ref_full"],
        "k--",
        linewidth=1.4,
        label="Reference path"
    )

    ax.plot(
        results_no_sat["x_global_history"],
        results_no_sat["y_global_history"],
        linewidth=1.7,
        label="Friction step, saturation disabled"
    )

    ax.plot(
        results_sat["x_global_history"],
        results_sat["y_global_history"],
        linewidth=1.7,
        label="Friction step, saturation enabled"
    )

    ax.set_title("Friction Disturbance: Vehicle Trajectory")
    ax.set_xlabel("X position [m]")
    ax.set_ylabel("Y position [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")

    # ------------------------------------------------------------
    # 2. Lateral tracking error
    # ------------------------------------------------------------
    ax = axes[1]

    ax.plot(
        time_no,
        ey_no,
        linewidth=1.7,
        label=r"Saturation disabled: $e_y$"
    )

    ax.plot(
        time_sat,
        ey_sat,
        linewidth=1.7,
        label=r"Saturation enabled: $e_y$"
    )

    ax.axhline(ey_limit, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(-ey_limit, color="black", linestyle="--", linewidth=1.0)

    ax.set_title(r"Lateral Tracking Error Under Reduced Friction")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$e_y$ [m]")
    ax.grid(True)
    ax.legend(loc="best")

    # ------------------------------------------------------------
    # 3. Steering input
    # ------------------------------------------------------------
    ax = axes[2]

    ax.plot(
        time_no,
        delta_no_deg,
        linewidth=1.7,
        label="Saturation disabled"
    )

    ax.plot(
        time_sat,
        delta_sat_deg,
        linewidth=1.7,
        label="Saturation enabled"
    )

    ax.axhline(steering_limit_deg, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(-steering_limit_deg, color="black", linestyle="--", linewidth=1.0)

    ax.set_title("Controller Steering Input")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\delta$ [deg]")
    ax.grid(True)
    ax.legend(loc="best")

    # ------------------------------------------------------------
    # 4. Tyre utilisation and friction coefficient
    # ------------------------------------------------------------
    ax1 = axes[3]
    ax2 = ax1.twinx()

    ax1.plot(
        time_no,
        front_no,
        linewidth=1.5,
        linestyle="-",
        label="No saturation: front utilisation"
    )

    ax1.plot(
        time_no,
        rear_no,
        linewidth=1.5,
        linestyle="--",
        label="No saturation: rear utilisation"
    )

    ax1.plot(
        time_sat,
        front_sat,
        linewidth=1.7,
        linestyle="-",
        label="Saturation enabled: front utilisation"
    )

    ax1.plot(
        time_sat,
        rear_sat,
        linewidth=1.7,
        linestyle="--",
        label="Saturation enabled: rear utilisation"
    )

    ax1.axhline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Tyre saturation threshold"
    )

    ax2.plot(
        time_sat,
        mu_sat,
        color="black",
        linewidth=1.4,
        alpha=0.6,
        label=r"$\mu$"
    )

    ax1.set_title("Tyre Utilisation and Friction Step")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Tyre force utilisation [-]")
    ax2.set_ylabel(r"Coefficient of friction $\mu$ [-]")

    ax1.grid(True)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()