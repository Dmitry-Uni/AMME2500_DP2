"""Controller entry for 2_State_Model.

This module supports being run directly. When executed as a script it will
ensure the `control` package directory is on `sys.path` so sibling modules
like `Vehicle_Params` can be imported.
"""
import os
import sys
import numpy as np

try:
    # Prefer package-relative imports when available
    from . import Reference_States  # type: ignore
except Exception:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import Reference_States  # type: ignore

try:
    # Prefer package-relative imports when available
    from . import Vehicle_Params  # type: ignore
except Exception:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import Vehicle_Params  # type: ignore


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def error_states(pos, psi, pos_ref, psi_ref):
    X, Y = pos
    X_ref, Y_ref = pos_ref

    dx = X - X_ref
    dy = Y - Y_ref

    # Signed lateral error in the reference path frame
    e_y = -np.sin(psi_ref) * dx + np.cos(psi_ref) * dy

    # Wrapped heading error
    e_psi = wrap_to_pi(psi - psi_ref)

    return np.array([e_y, e_psi])

def feedback_control(e_y, e_psi):
    # Simple proportional controller for demonstration
    k_y = 0.5  # Lateral error gain
    k_phi = 1.0  # Heading error gain

    delta_fb = -k_y * e_y - k_phi * e_psi
    return delta_fb

def feedforward_control(curvature):
    return np.arctan(Vehicle_Params.whlb * curvature)

def total_control(e_y, e_psi, curvature):
    delta_fb = feedback_control(e_y, e_psi)
    delta_ff = feedforward_control(curvature)
    return delta_fb + delta_ff

def main():
    print("Running Controller main() \n")
    # Example usage: call Reference_States.init_path() if available
    if hasattr(Reference_States, 'init_path'):
        Reference_States.init_path()

    pos = np.array([10, 4])
    psi = 0.1  # Example heading angle in radians

    nearest_x, nearest_y, curvature, heading = Reference_States.nearest_ref(pos[0], pos[1])
    print("Nearest reference state for (10, 4):", nearest_x, nearest_y, curvature, heading)

    pos_ref = np.array([nearest_x, nearest_y])
    psi_ref = heading

    x = np.array(error_states(pos, psi, pos_ref, psi_ref))
    print("Error states for (10, 4):", x)


if __name__ == '__main__':
    main()
