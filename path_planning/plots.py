import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation


def _obstacle_patch(obstacle, t=0.0, **kwargs):
    if obstacle.shape == 'circle':
        return patches.Circle(obstacle.center_at(t), obstacle.radius, **kwargs)

    cx, cy = obstacle.center_at(t)
    rect = patches.Rectangle(
        (-obstacle.width / 2, -obstacle.height / 2),
        obstacle.width,
        obstacle.height,
        **kwargs
    )
    trans = transforms.Affine2D().rotate(obstacle.angle).translate(cx, cy)
    rect.set_transform(trans + plt.gca().transData)
    return rect


def _vehicle_patch(position, heading, length=5.0, width=2.5, **kwargs):
    rect = patches.Rectangle((-length / 2, -width / 2), length, width, **kwargs)
    trans = transforms.Affine2D().rotate(heading).translate(position[0], position[1])
    rect.set_transform(trans + plt.gca().transData)
    return rect


def plot_path(sol, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    path = sol.get_path()
    print("Path: ", path)
    path_line, = ax.plot(path[:, 0], path[:, 1], **kwargs)
    return path_line


def update_path(sol, path_line):
    path = sol.get_path()
    path_line.set_xdata(path[:, 0])
    path_line.set_ydata(path[:, 1])
    fig = plt.gcf()
    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_environment(environment, ax=None, t=0.0, obstacles_style=None, start_style=None, goal_style=None):
    if ax is None:
        ax = plt.gca()

    obstacles_style = {} if obstacles_style is None else dict(obstacles_style)
    start_style = {} if start_style is None else dict(start_style)
    goal_style = {} if goal_style is None else dict(goal_style)

    ax.set_aspect('equal', adjustable='box')

    obstacles_style.setdefault('color', 'k')
    obstacles_style.setdefault('alpha', 0.35)
    for obstacle in environment.obstacles:
        ax.add_patch(_obstacle_patch(obstacle, t=t, **obstacles_style))

    start_style.setdefault('color', 'r')
    start_style.setdefault('markersize', 12)
    ax.plot(environment.start[0], environment.start[1], 's', **start_style)

    goal_style.setdefault('color', 'g')
    goal_style.setdefault('markersize', 12)
    ax.plot(environment.goal[0], environment.goal[1], 's', **goal_style)

    ax.set_xlim([0, environment.width])
    ax.set_ylim([0, environment.height])


def interpolate_path(path, times, t):
    x = np.interp(t, times, path[:, 0])
    y = np.interp(t, times, path[:, 1])
    return np.array([x, y])


def heading_from_path(path, idx):
    if idx <= 0:
        diff = path[1] - path[0]
    elif idx >= len(path) - 1:
        diff = path[-1] - path[-2]
    else:
        diff = path[idx + 1] - path[idx - 1]
    return np.arctan2(diff[1], diff[0])


def animate_solution(sol, interval=80, vehicle_length=4.9, vehicle_width=1.94,
                     save_path=None, show=True):
    """Animate the final vehicle path and the moving obstacles."""
    env = sol.environment
    path = sol.get_path()
    times = env.path_times(path)
    total_time = times[-1]

    fig, ax = plt.subplots(figsize=(7, 7))
    num_frames = len(path)

    def draw(frame):
        ax.clear()
        t = times[frame]
        plot_environment(env, ax=ax, t=t)
        ax.plot(path[:, 0], path[:, 1], '--', linewidth=1.5, label='final planned path')
        ax.plot(path[:frame + 1, 0], path[:frame + 1, 1], linewidth=2.5, label='vehicle trace')

        vehicle_position = path[frame]
        heading = heading_from_path(path, frame)
        ax.add_patch(_vehicle_patch(
            vehicle_position,
            heading,
            length=vehicle_length,
            width=vehicle_width,
            color='tab:blue',
            alpha=0.85,
        ))

        ax.set_title(f"Dynamic obstacle avoidance, t = {t:.1f} s / {total_time:.1f} s")
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.grid(True)
        ax.legend(loc='upper left')
        return []

    anim = FuncAnimation(fig, draw, frames=num_frames, interval=interval, blit=False)

    if save_path:
        anim.save(save_path)

    if show:
        plt.show()

    return anim
