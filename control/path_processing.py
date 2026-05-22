import numpy as np
import matplotlib.pyplot as plt


def evaluate_spline_coeffs(coeffs, breaks, u_eval):
    """
    Evaluate a 2D piecewise cubic spline from SciPy CubicSpline-style coefficients.

    coeffs shape: (4, n_segments, 2)
        coeffs[0, i] -> cubic coefficient for segment i
        coeffs[1, i] -> quadratic coefficient
        coeffs[2, i] -> linear coefficient
        coeffs[3, i] -> constant coefficient

    breaks shape: (n_segments + 1,)
        spline breakpoints used when the original CubicSpline was created

    u_eval:
        parameter values where path is evaluated
    """

    coeffs = np.asarray(coeffs)
    breaks = np.asarray(breaks)
    u_eval = np.asarray(u_eval)

    n_segments = coeffs.shape[1]

    path = np.zeros((len(u_eval), 2))
    dpath = np.zeros((len(u_eval), 2))
    ddpath = np.zeros((len(u_eval), 2))

    for j, u in enumerate(u_eval):
        # Find active spline segment
        i = np.searchsorted(breaks, u, side="right") - 1
        i = np.clip(i, 0, n_segments - 1)

        h = u - breaks[i]

        c0 = coeffs[0, i]   # cubic
        c1 = coeffs[1, i]   # quadratic
        c2 = coeffs[2, i]   # linear
        c3 = coeffs[3, i]   # constant

        # Position
        path[j] = c0*h**3 + c1*h**2 + c2*h + c3

        # First derivative wrt spline parameter u
        dpath[j] = 3*c0*h**2 + 2*c1*h + c2

        # Second derivative wrt spline parameter u
        ddpath[j] = 6*c0*h + 2*c1

    return path, dpath, ddpath


def curvature_and_radius_from_coeffs(coeffs, breaks, num_points=500):
    """
    Returns path, curvature and radius of curvature along the spline.
    """

    u_eval = np.linspace(breaks[0], breaks[-1], num_points)

    path, dpath, ddpath = evaluate_spline_coeffs(coeffs, breaks, u_eval)

    dx = dpath[:, 0]
    dy = dpath[:, 1]
    ddx = ddpath[:, 0]
    ddy = ddpath[:, 1]

    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2)**1.5

    # Avoid division by zero
    eps = 1e-12
    curvature = numerator / np.maximum(denominator, eps)

    # Radius of curvature
    radius = np.full_like(curvature, np.inf)
    nonzero = np.abs(curvature) > eps
    radius[nonzero] = 1 / np.abs(curvature[nonzero])

    return u_eval, path, curvature, radius