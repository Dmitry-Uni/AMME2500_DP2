# Dynamic Rectangle Obstacle Path Planning Adaptation

This is an adapted version of the uploaded PSO path-planning project. It adds:

1. Rectangular obstacles for vehicle-shaped obstacles.
2. Moving obstacles using constant velocity.
3. Time-aware path collision checking.
4. Animation of a vehicle travelling along the final PSO path while the obstacles move.

## Key idea

The original project checks whether each path point intersects a static circle. This version estimates the time at which the vehicle reaches each point:

```python
time = cumulative_path_distance / vehicle_speed
```

Then each obstacle is moved to its position at that same time:

```python
obstacle_center(t) = obstacle_center(0) + obstacle_velocity * t
```

The PSO cost function penalises paths that collide with obstacles at the corresponding time.

## Run

```bash
python main_dynamic.py
```

## Example obstacle definition

```python
{
    'shape': 'rectangle',
    'center': [62, 72],
    'width': 14,
    'height': 6,
    'angle': -10,
    'velocity': [-0.3, -1.8],
    'name': 'moving_vehicle',
}
```

## Important limitation

This is a path generation layer, not a full vehicle controller. For the design project, the clean framing is:

- PSO/spline planner generates a collision-free reference path.
- A state-space vehicle model and controller track that reference path.
- The moving obstacle model is used as a numerical validation scenario.
