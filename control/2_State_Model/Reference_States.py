# Import vehicle parameters from the sibling `control/Vehicle_Params.py`.
# We prefer a package-relative import, but also support running this file
# directly as a script by adding the parent `control` directory to `sys.path`.
import os
import sys
import numpy as np

try:
	from .. import Vehicle_Params  # type: ignore
except Exception:
	parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	if parent_dir not in sys.path:
		sys.path.insert(0, parent_dir)
	import Vehicle_Params

try:
	from ...control import main_dynamic  # type: ignore
except Exception:
    grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if grandparent_dir not in sys.path:
        sys.path.insert(0, grandparent_dir)
    import main_dynamic  # type: ignore


### Simplified Kinematic Bicycle Model for Path Following Control
### Vehicle parameters are defined in control/Vehicle_Params.py

# Returns error states for a given reference path and current vehicle state
def main():
    print("Running References main()")
    # Example usage: call a function from main_dynamic.py if available
    if hasattr(main_dynamic, 'final_path_details'):
        u, path, curvature, radius = main_dynamic.final_path_details()
        print("Final path details obtained from main_dynamic.py")
        print("Path Curvature:", curvature.shape, curvature[:5])
        print("Polynomial coefficient (u): ", u.shape, u[:5])  # Print first 5 points of the path


if __name__ == '__main__':
	main()