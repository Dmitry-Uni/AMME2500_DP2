import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import Controller
import control.Reference_States as Reference_States

try:
    from . import Vehicle_Params  # type: ignore
except Exception:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import Vehicle_Params  # type: ignore


def init_state_matrices():
    """
    2-state linearised kinematic bicycle error model:

        x = [e_y, e_psi]^T

        e_y_dot   = V_x e_psi
        e_psi_dot = (V_x / L) delta - V_x kappa_ref
    """
    A = np.array([
        [0.0, Vehicle_Params.V_x],
        [0.0, 0.0]
    ])

    B = np.array([
        [0.0],
        [Vehicle_Params.V_x / Vehicle_Params.whlb]
    ])

    E = np.array([
        [0.0],
        [-Vehicle_Params.V_x]
    ])

    return A, B, E


def init_state(e_y0=0.8, e_psi0=np.radians(3.0)):
    """
    Initial state: [lateral error, heading error]^T.

    Non-zero initial error is useful for report plots because it shows
    convergence back toward the reference path.
    """
    return np.array([
        [e_y0],
        [e_psi0]
    ])


def compute_path_arclength(path):
    """
    Compute cumulative arc length along the reference path.
    """
    diffs = np.diff(path, axis=0)
    ds = np.linalg.norm(diffs, axis=1)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    return s


def reconstruct_vehicle_position(path_ref, heading_ref, x_history):
    """
    Reconstruct global vehicle position from lateral error.

    This is appropriate for the 2-state error-coordinate model.
    """
    e_y = x_history[:, 0]

    normal_ref = np.column_stack((
        -np.sin(heading_ref),
        np.cos(heading_ref)
    ))

    vehicle_pos = path_ref + e_y[:, None] * normal_ref
    return vehicle_pos


def simulate_2_state(path, curvature, heading):
    """
    Simulate the 2-state linearised kinematic error model along the path.
    """
    A, B, E = init_state_matrices()
    x = init_state()

    n_steps = len(path)

    s = compute_path_arclength(path)
    t = s / Vehicle_Params.V_x

    x_history = np.zeros((n_steps, 2))
    delta_history = np.zeros(n_steps)

    x_history[0, :] = x[:, 0]

    for k in range(1, n_steps):
        dt = t[k] - t[k - 1]

        kappa_ref = curvature[k]
        e_y = x[0, 0]
        e_psi = x[1, 0]

        delta = Controller.total_control(e_y, e_psi, kappa_ref)
        delta = float(np.asarray(delta).squeeze())

        # Steering saturation
        delta = np.clip(
            delta,
            -Vehicle_Params.max_steering_angle,
            Vehicle_Params.max_steering_angle
        )

        x_dot = A @ x + B * delta + E * kappa_ref
        x = x + dt * x_dot

        x_history[k, :] = x[:, 0]
        delta_history[k] = delta

    vehicle_pos = reconstruct_vehicle_position(path, heading, x_history)

    return t, vehicle_pos, x_history, delta_history


def plot_path_tracking(path, vehicle_pos, save_path=None):
    plt.figure(figsize=(9, 4.5))

    plt.plot(path[:, 0], path[:, 1], '--', linewidth=2.0, label='Reference path')
    plt.plot(vehicle_pos[:, 0], vehicle_pos[:, 1], linewidth=2.0, label='2-state vehicle trajectory')

    plt.scatter(path[0, 0], path[0, 1], marker='o', label='Start')
    plt.scatter(path[-1, 0], path[-1, 1], marker='x', label='Goal')

    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('Global X position [m]')
    plt.ylabel('Global Y position [m]')
    plt.title('Reference Path and 2-State Vehicle Tracking Response')
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_tracking_errors(t, x_history, save_path=None):
    e_y = x_history[:, 0]
    e_psi = np.degrees(x_history[:, 1])

    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax[0].plot(t, e_y, linewidth=2.0)
    ax[0].grid(True)
    ax[0].set_ylabel('Lateral error [m]')
    ax[0].set_title('2-State Path Tracking Error Response')

    ax[1].plot(t, e_psi, linewidth=2.0)
    ax[1].grid(True)
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Heading error [deg]')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_steering_input(t, delta_history, save_path=None):
    plt.figure(figsize=(8, 3.5))

    plt.plot(t, np.degrees(delta_history), linewidth=2.0)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Steering angle [deg]')
    plt.title('Controller Steering Input')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_mu_required(t, curvature, save_path=None):
    mu_required = (Vehicle_Params.V_x ** 2) * np.abs(curvature) / 9.81

    plt.figure(figsize=(8, 3.5))

    plt.plot(t, mu_required, linewidth=2.0)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Required friction coefficient, mu')
    plt.title('Required Friction Coefficient Along Path')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def main():
    print("Running 2-state simulation...\n")

    u, path, curvature, radius, heading = Reference_States.init_path()

    t, vehicle_pos, x_history, delta_history = simulate_2_state(
        path,
        curvature,
        heading
    )

    output_dir = os.path.dirname(__file__)

    plot_path_tracking(
        path,
        vehicle_pos,
        save_path=os.path.join(output_dir, 'path_tracking_overlay.png')
    )

    plot_tracking_errors(
        t,
        x_history,
        save_path=os.path.join(output_dir, 'tracking_errors.png')
    )

    plot_steering_input(
        t,
        delta_history,
        save_path=os.path.join(output_dir, 'steering_input.png')
    )

    plot_mu_required(
        t,
        curvature,
        save_path=os.path.join(output_dir, 'required_mu.png')
    )

if __name__ == '__main__':
    main()