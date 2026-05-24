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
    

def init_state_matricies():
    A = np.array([[0, Vehicle_Params.V_x], [0, 0]])
    B = np.array([[0], [Vehicle_Params.V_x / Vehicle_Params.whlb]])
    E = np.array([[0], [- Vehicle_Params.V_x]])
    return A, B, E

def init_state():
    # Initial state: [lateral error, heading error]
    return np.array([[0], [0]])

def main():
    print("Running Simulation main() \n")
    u, path, curvature, radius, heading = Reference_States.init_path()
    print("Path details in Simulation main():")
    print("u shape:", u.shape, "path shape:", path.shape, "curvature shape:", curvature.shape, "radius shape:", radius.shape, "heading shape:", heading.shape)


if __name__ == '__main__':
    main()