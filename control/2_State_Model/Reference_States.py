# Import vehicle parameters from the sibling `control/Vehicle_Params.py`.
# We prefer a package-relative import, but also support running this file
# directly as a script by adding the parent `control` directory to `sys.path`.
import os
import sys
import numpy as np

try:
    from ... import main_dynamic  # type: ignore
except Exception:
    grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if grandparent_dir not in sys.path:
        sys.path.insert(0, grandparent_dir)
    import main_dynamic  # type: ignore


### Simplified Kinematic Bicycle Model for Path Following Control
### Vehicle parameters are defined in control/Vehicle_Params.py

# Returns error states for a given reference path and current vehicle state

def load_path_from_main():
    if not hasattr(main_dynamic, 'final_path_details'):
        raise AttributeError('main_dynamic does not expose final_path_details')

    result = main_dynamic.final_path_details()
    if len(result) != 5:
        raise ValueError('final_path_details() must return (u, path, curvature, radius, heading)')

    u, path, curvature, radius, heading = result
    path = np.asarray(path)
    curvature = np.asarray(curvature)
    heading = np.asarray(heading)

    if path.size == 0:
        raise ValueError('Path data is empty')

    return u, path, curvature, radius, heading


def init_path():
    u, path, curvature, radius, heading = load_path_from_main()
    print('Final path details obtained from main_dynamic.py \n')
    return u, path, curvature, radius, heading


u, path, curvature, radius, heading = init_path()

def find_nearest(array, value):
    array = np.asarray(array)
    if array.size == 0:
        raise ValueError('Input array is empty')
    idx = int(np.abs(array - value).argmin())
    return idx


def nearest_ref(X, Y):
    if path.size == 0:
        raise ValueError('Path array is empty')

    distances = np.linalg.norm(path - np.array([X, Y]), axis=1)
    idx = int(np.argmin(distances))
    nearest_x = path[:,0][idx]
    nearest_y = path[:,1][idx]

    return nearest_x, nearest_y, curvature[idx], heading[idx]

def path_size():
    return path.shape[0]
