import matplotlib.pyplot as plt
from control import Vehicle_Params
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
    'vehicle_speed': Vehicle_Params.V_x,
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

def generate_path():
    bestsol, pop = PSO(problem, **pso_params)

    coeff_pack = bestsol['details']['sol'].get_path('coeffs')
    coeffs = np.asarray(coeff_pack[0])
    breaks = np.asarray(coeff_pack[1])

    return bestsol, pop, coeffs, breaks

def final_path_details(*args):
    bestsol, pop, coeffs, breaks = generate_path()

    u, path, curvature, radius, heading = curvature_and_radius_from_coeffs(
        coeffs,
        breaks,
        num_points=500
    )
    if 'bestsol' in args:
        return u, path, curvature, radius, heading, bestsol
    
    else:
        return u, path, curvature, radius, heading

def final_path_length_and_time(bestsol):
    return bestsol['details']['length'], bestsol['details']['times'][-1]


def print_final_solution_details(bestsol):
    # Print final solution details
    print('\nFinal best solution')
    print('Cost:', bestsol['cost'])
    print('Path length:', bestsol['details']['length'])
    print('Total Time:', bestsol['details']['times'][-1])
    print('Minimum dynamic clearance:', bestsol['details']['min_clearance'])
    print('Collision violations:', bestsol['details']['collision_violation_count'], '\n')

if __name__ == '__main__':
    bestsol, pop, coeffs, breaks = generate_path()
    print("Final best solution")


'''
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

# Path with dual y-axes for heading angle
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]', color=color)
ax1.plot(path[:, 0], path[:, 1], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('heading angle, psi [rad]', color=color)  # we already handled the x-label with ax1
ax2.plot(path[:, 0], heading, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# Path with dual y-axes for curvature
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]', color=color)
ax1.plot(path[:, 0], path[:, 1], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('curvature, kappa [1/m]', color=color)  # we already handled the x-label with ax1
ax2.plot(path[:, 0], curvature, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
'''

'''
# heading angle
plt.figure()
plt.plot(path[:, 0], heading, linewidth=1.8)
plt.grid(True)
plt.xlabel("x [m]")
plt.ylabel("Heading angle, psi [rad]")
plt.title("Heading Angle Along Spline")
plt.show()

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