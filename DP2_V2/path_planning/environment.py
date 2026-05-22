import numpy as np


class Environment:
    """2D path-planning environment with static or moving obstacles."""

    def __init__(self, width=100, height=100, robot_radius=0, obstacles=None,
                 start=None, goal=None, vehicle_speed=8.0):
        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        self.obstacles = [] if obstacles is None else obstacles
        self.start = np.array(start, dtype=float) if start is not None else None
        self.goal = np.array(goal, dtype=float) if goal is not None else None
        self.vehicle_speed = vehicle_speed

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def add_obstacles(self, obstacles):
        self.obstacles.extend(obstacles)

    def clear_obstacles(self):
        self.obstacles = []

    def in_collision(self, point, t=0.0):
        point = np.array(point, dtype=float)
        for obstacle in self.obstacles:
            if obstacle.in_collision(point, self.robot_radius, t=t):
                return True
        return False

    def path_in_collision(self, path, vehicle_speed=None):
        times = self.path_times(path, vehicle_speed=vehicle_speed)
        for point, t in zip(path, times):
            if self.in_collision(point, t=t):
                return True
        return False

    def in_environment(self, point):
        point = np.array(point, dtype=float)
        min_x = self.robot_radius
        max_x = self.width - self.robot_radius
        min_y = self.robot_radius
        max_y = self.height - self.robot_radius
        return (min_x <= point[0] <= max_x) and (min_y <= point[1] <= max_y)

    def path_in_environment(self, path):
        return all(self.in_environment(point) for point in path)

    def clip_point(self, point):
        point = np.array(point, dtype=float)
        min_x = self.robot_radius
        max_x = self.width - self.robot_radius
        min_y = self.robot_radius
        max_y = self.height - self.robot_radius
        return np.array([np.clip(point[0], min_x, max_x), np.clip(point[1], min_y, max_y)])

    def clip_path(self, path):
        return np.array([self.clip_point(point) for point in path])

    def in_goal(self, point):
        return np.linalg.norm(np.array(point, dtype=float) - self.goal) <= self.robot_radius

    def path_in_goal(self, path):
        return self.in_goal(path[-1])

    def in_start(self, point):
        return np.linalg.norm(np.array(point, dtype=float) - self.start) <= self.robot_radius

    def path_in_start(self, path):
        return self.in_start(path[0])

    def path_length(self, path):
        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            return 0.0
        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return float(np.sum(segment_lengths))

    def path_times(self, path, vehicle_speed=None):
        """
        Assign an arrival time to every point on the path.

        This is the key addition for moving obstacles. The planner assumes the
        vehicle follows the spline at approximately constant speed, so time is
        cumulative path distance divided by vehicle speed.
        """
        path = np.asarray(path, dtype=float)
        speed = self.vehicle_speed if vehicle_speed is None else vehicle_speed
        speed = max(float(speed), 1e-6)
        if len(path) < 2:
            return np.array([0.0])
        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cumulative_distance = np.insert(np.cumsum(segment_lengths), 0, 0.0)
        return cumulative_distance / speed

    def count_violations(self, path, vehicle_speed=None):
        violations = 0
        times = self.path_times(path, vehicle_speed=vehicle_speed)
        details = {
            'start_violation': False,
            'goal_violation': False,
            'environment_violation': False,
            'environment_violation_count': 0,
            'collision_violation': False,
            'collision_violation_count': 0,
            'min_clearance': np.inf,
        }

        if not self.path_in_start(path):
            violations += 1
            details['start_violation'] = True

        if not self.path_in_goal(path):
            violations += 1
            details['goal_violation'] = True

        for point, t in zip(path, times):
            if not self.in_environment(point):
                violations += 1
                details['environment_violation_count'] += 1

            for obstacle in self.obstacles:
                clearance = obstacle.clearance(point, self.robot_radius, t=t)
                details['min_clearance'] = min(details['min_clearance'], clearance)
                if clearance <= 0:
                    violations += 1
                    details['collision_violation_count'] += 1

        details['environment_violation'] = details['environment_violation_count'] > 0
        details['collision_violation'] = details['collision_violation_count'] > 0
        return violations, details

    def path_is_valid(self, path, vehicle_speed=None):
        violations, _ = self.count_violations(path, vehicle_speed=vehicle_speed)
        return violations == 0


class Obstacle:
    """
    Obstacle class supporting circular and rectangular obstacles.

    Parameters
    ----------
    shape : 'circle' or 'rectangle'
    center : initial [x, y] position
    velocity : [vx, vy], optional. Use [0, 0] for static obstacles.
    radius : circle radius
    width, height : rectangle dimensions
    angle : rectangle heading in degrees, measured counter-clockwise from +x
    """

    def __init__(self, center, radius=None, shape='circle', width=None, height=None,
                 angle=0.0, velocity=None, name=None):
        self.shape = shape.lower()
        self.center = np.array(center, dtype=float)
        self.velocity = np.array([0.0, 0.0] if velocity is None else velocity, dtype=float)
        self.radius = radius
        self.width = width
        self.height = height
        self.angle = np.deg2rad(angle)
        self.name = name or self.shape

        if self.shape == 'circle' and self.radius is None:
            raise ValueError("Circle obstacles require a radius.")
        if self.shape == 'rectangle' and (self.width is None or self.height is None):
            raise ValueError("Rectangle obstacles require width and height.")
        if self.shape not in ['circle', 'rectangle']:
            raise ValueError("shape must be either 'circle' or 'rectangle'.")

    def center_at(self, t=0.0):
        return self.center + self.velocity * float(t)

    def _point_in_obstacle_frame(self, point, t=0.0):
        p = np.array(point, dtype=float) - self.center_at(t)
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        return np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1]])

    def clearance(self, point, robot_radius=0.0, t=0.0):
        """
        Signed clearance from the obstacle boundary.

        Positive = safe gap, zero = touching, negative = collision.
        The robot_radius inflates the obstacle to give a safety buffer.
        """
        point = np.array(point, dtype=float)

        if self.shape == 'circle':
            return float(np.linalg.norm(point - self.center_at(t)) - (self.radius + robot_radius))

        local = self._point_in_obstacle_frame(point, t=t)
        half_extents = np.array([self.width / 2.0 + robot_radius,
                                 self.height / 2.0 + robot_radius])
        q = np.abs(local) - half_extents
        outside_distance = np.linalg.norm(np.maximum(q, 0.0))
        inside_distance = min(max(q[0], q[1]), 0.0)
        return float(outside_distance + inside_distance)

    def in_collision(self, point, robot_radius=0.0, t=0.0):
        return self.clearance(point, robot_radius=robot_radius, t=t) <= 0
