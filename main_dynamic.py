import matplotlib.pyplot as plt
from control.path_processing import curvature_and_radius_from_coeffs
import path_planning as pp
from pso import PSO
import numpy as np

plt.rcParams["figure.autolayout"] = True

# Create environment
# vehicle_speed is map-units/second. This is used to synchronize the ego vehicle
# with the moving obstacles during collision checking.
env_params = {
    'width': 100,
    'height': 8,
    'robot_radius': 1.94,  # Approximate radius of a typical car (for collision checking)
    'vehicle_speed': 8.0,
    'start': [5, 2.6],
    'goal': [95, 2.6],
}
env = pp.Environment(**env_params)

# Rectangular vehicle-like obstacles.
# velocity = [vx, vy]. Use [0, 0] for static obstacles.
obstacles = [
    {
        'shape': 'rectangle',
        'center': [40, 2.5],
        'width': 4.9,
        'height': 1.94,
        'angle': 0,
        'velocity': [0, 0],
        'name': 'parked_vehicle',
    },
    {
        'shape': 'rectangle',
        'center': [50, 2.5],
        'width': 4.9,
        'height': 1.94,
        'angle': 0,
        'velocity': [0, 0],
        'name': 'moving_vehicle',
    },
]

for obs in obstacles:
    env.add_obstacle(pp.Obstacle(**obs))

# Create cost function
num_control_points = 5
resolution = 90
cost_function = pp.EnvCostFunction(env, num_control_points, resolution)

# Optimization Problem
problem = {
    'num_var': 2 * num_control_points,
    'var_min': 0,
    'var_max': 1,
    'cost_function': cost_function,
}

# Callback function for live PSO path updates
path_line = None

def callback(data):
    global path_line
    it = data['it']
    sol = data['gbest']['details']['sol']
    length = data['gbest']['details']['length']
    min_clearance = data['gbest']['details']['min_clearance']

    if it == 1:
        plt.figure(figsize=[7, 7])
        pp.plot_environment(env, t=0.0)
        path_line = pp.plot_path(sol, color='b', linewidth=2)
        plt.grid(True)
        plt.show(block=False)
    else:
        pp.update_path(sol, path_line)

    plt.title(f"Iteration: {it}, Length: {length:.2f}, Min clearance: {min_clearance:.2f}")


# Run PSO
pso_params = {
    'max_iter': 20,
    'pop_size': 140,
    'c1': 2,
    'c2': 1,
    'w': 0.8,
    'wdamp': 1,
    'resetting': 30,
}

#bestsol, pop = PSO(problem, callback=callback, **pso_params)
bestsol, pop = PSO(problem, **pso_params)

coeff_pack = bestsol['details']['sol'].get_path('coeffs')

coeffs = np.asarray(coeff_pack[0])
breaks = np.asarray(coeff_pack[1])

u, path, curvature, radius = curvature_and_radius_from_coeffs(
    coeffs,
    breaks,
    num_points=500
)


# Print final solution details
final_sol = bestsol['details']['sol']
print('\nFinal best solution')
print('Cost:', bestsol['cost'])
print('Path length:', bestsol['details']['length'])
print('Minimum dynamic clearance:', bestsol['details']['min_clearance'])
print('Collision violations:', bestsol['details']['collision_violation_count'])

def final_path_details():
    coeff_pack = bestsol['details']['sol'].get_path('coeffs')
    coeffs = np.asarray(coeff_pack[0])
    breaks = np.asarray(coeff_pack[1])
    u, path, curvature, radius = curvature_and_radius_from_coeffs(
        coeffs,
        breaks,
        num_points=500
    )
    return u, path, curvature, radius


# Visualizations

# Animate the vehicle following the final path while obstacles move.
# To save a GIF, use: save_path='dynamic_obstacle_avoidance.gif'
#anim = pp.animate_solution(final_sol, interval=80, vehicle_length=5.0, vehicle_width=2.5)

#path
plt.figure()
plt.plot(path[:, 0], path[:, 1], linewidth=1.8)
plt.axis("equal")
plt.grid(True)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Spline Path")
plt.show()

'''
#curvature
plt.figure()
plt.plot(u, curvature, linewidth=1.8)
plt.grid(True)
plt.xlabel("Spline parameter, u")
plt.ylabel("Signed curvature, kappa [1/m]")
plt.title("Path Curvature Along Spline")
plt.show()

#radius
plt.figure()
plt.plot(u, radius, linewidth=1.8)
plt.grid(True)
plt.xlabel("Spline parameter, u")
plt.ylabel("Radius of curvature, R [m]")
plt.title("Radius of Curvature Along Path")
plt.ylim(0, np.nanpercentile(radius[np.isfinite(radius)], 95))
plt.show()

plt.close('all')
'''

# End of main_dynamic.py