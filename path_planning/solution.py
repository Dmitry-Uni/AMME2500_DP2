import numpy as np
from scipy.interpolate import CubicSpline

class SplinePath:
    """Class to represent a path as a cubic spline."""

    # Constructor
    def __init__(self, environment, control_points=[], resolution=100):
        self.environment = environment
        self.control_points = control_points
        self.resolution = resolution
        
    # Create random control points
    @staticmethod
    def random(environment, num_control_points=10, resolution=100):
        control_points = np.random.rand(num_control_points, 2) * np.array([environment.width, environment.height])
        return SplinePath(environment, control_points, resolution)
    
    # Create control points from list
    @staticmethod
    def from_list(environment, xy, resolution=100, normalized=False):
        control_points = np.array(xy).reshape(-1, 2)
        if normalized:
            control_points[:,0] *= environment.width
            control_points[:,1] *= environment.height
            
        return SplinePath(environment, control_points, resolution)

    # Get path
    def get_path(self, *args):
        
        # Add start and goal to control points
        start = self.environment.start
        goal = self.environment.goal
        points = np.vstack((start, self.control_points, goal))

        # Create spline
        t = np.linspace(0, 1, len(points))
        # Desired vehicle headings at start and goal
        start_heading = 0.0   # radians, 0 means pointing in +x direction
        goal_heading  = 0.0   # radians, return to +x direction

        # Tangent magnitude controls how strongly the spline follows the boundary heading
        tangent_scale = np.linalg.norm(goal - start)

        start_tangent = tangent_scale * np.array([
            np.cos(start_heading),
            np.sin(start_heading)
        ])

        goal_tangent = tangent_scale * np.array([
            np.cos(goal_heading),
            np.sin(goal_heading)
        ])

        cs = CubicSpline(
            t,
            points,
            bc_type=((1, start_tangent), (1, goal_tangent))
        )

        # Get path
        tt = np.linspace(0, 1, self.resolution)
        path = cs(tt)

        # Clip path to environment
        path = self.environment.clip_path(path)

        #print(f"CubicSpline coefficients: {cs.c}")  # Debug: print the coefficients of the cubic spline

        if "coeffs" in args:
            return [cs.c, tt]  # Return coefficients and parameter values for debugging

        return path
        

