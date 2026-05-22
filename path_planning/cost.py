from path_planning.solution import SplinePath
from path_planning.environment import Environment

START_VIOLATION_PENALTY = 1
GOAL_VIOLATION_PENALTY = 1
ENV_VIOLATION_PENALTY = 0.2
COLLISION_PENALTY = 1.5
LOW_CLEARANCE_PENALTY = 0.05


def PathPlanningCost(sol: SplinePath):
    path = sol.get_path()
    length = sol.environment.path_length(path)
    _, details = sol.environment.count_violations(path)

    cost = length

    if details['start_violation']:
        cost *= 1 + START_VIOLATION_PENALTY

    if details['goal_violation']:
        cost *= 1 + GOAL_VIOLATION_PENALTY

    if details['environment_violation']:
        cost *= 1 + details['environment_violation_count'] * ENV_VIOLATION_PENALTY

    if details['collision_violation']:
        cost *= 1 + details['collision_violation_count'] * COLLISION_PENALTY

    # Extra smooth penalty to encourage clearance even when there is no collision.
    # This makes PSO less likely to settle on paths that just scrape past obstacles.
    desired_clearance = 2.0 * sol.environment.robot_radius
    if details['min_clearance'] < desired_clearance:
        cost *= 1 + (desired_clearance - details['min_clearance']) * LOW_CLEARANCE_PENALTY

    details['sol'] = sol
    details['path'] = path
    details['times'] = sol.environment.path_times(path)
    details['length'] = length
    details['cost'] = cost

    return cost, details


def EnvCostFunction(environment: Environment, num_control_points=10, resolution=100):
    def CostFunction(xy):
        sol = SplinePath.from_list(environment, xy, resolution, normalized=True)
        return PathPlanningCost(sol)
    return CostFunction
