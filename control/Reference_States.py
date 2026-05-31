# Import vehicle parameters from the sibling `control/Vehicle_Params.py`.
# We prefer a package-relative import, but also support running this file
# directly as a script by adding the parent `control` directory to `sys.path`.
import os
import sys
import numpy as np
#from .. import path_generation

try:
    import main_dynamic
    import path_generation
except Exception:
    grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', '.'))
    if grandparent_dir not in sys.path:
        sys.path.insert(0, grandparent_dir)
    import main_dynamic  # type: ignore
    import path_generation  # type: ignore

def load_path_from_main():
    if not hasattr(main_dynamic, 'final_path_details'):
        raise AttributeError('main_dynamic does not expose final_path_details')

    result = main_dynamic.final_path_details('bestsol')
    if len(result) != 6:
        raise ValueError('final_path_details() must return (u, path, curvature, radius, heading, bestsol)')

    u, path, curvature, radius, heading, bestsol = result
    main_dynamic.print_final_solution_details(bestsol)  # Print solution details from main_dynamic
    path = np.asarray(path)
    curvature = np.asarray(curvature)
    heading = np.asarray(heading)

    if path.size == 0:
        raise ValueError('Path data is empty')

    return u, path, curvature, radius, heading, bestsol

'''
def init_path():
    u, path, curvature, radius, heading, bestsol = load_path_from_main()
    print('Final path details obtained from main_dynamic.py \n')
    length, time = main_dynamic.final_path_length_and_time(bestsol)
    return u, path, curvature, radius, heading, length, time
'''

def init_path():
    results = path_generation.load_path_details("generated_paths/path_008.npz")
    u = results["u"]
    path = results["path"]
    curvature = results["curvature"]
    radius = results["radius"]
    heading = results["heading"]
    bestsol = results["bestsol"]
    length, time = main_dynamic.final_path_length_and_time(bestsol)
    return u, path, curvature, radius, heading, length, time

def init_path_mellow():
    results = path_generation.load_path_details("generated_paths/path_005.npz")
    u = results["u"]
    path = results["path"]
    curvature = results["curvature"]
    radius = results["radius"]
    heading = results["heading"]
    bestsol = results["bestsol"]
    length, time = main_dynamic.final_path_length_and_time(bestsol)
    return u, path, curvature, radius, heading, length, time

## Temporary global variables to debug
#u, path, curvature, radius, heading = init_path()

def find_nearest(array, value):
    array = np.asarray(array)
    if array.size == 0:
        raise ValueError('Input array is empty')
    idx = int(np.abs(array - value).argmin())
    return idx


def nearest_ref(X, Y, path_ref, curvature_ref, heading_ref):
    distances = np.linalg.norm(path_ref - np.array([X, Y]), axis=1)
    idx = int(np.argmin(distances))

    nearest_x = path_ref[idx, 0]
    nearest_y = path_ref[idx, 1]

    return nearest_x, nearest_y, curvature_ref[idx], heading_ref[idx]

def path_size(path):
    return path.shape[0]
