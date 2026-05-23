import numpy as np

### Vehicle Parameters - Tesla Model 3

    #Vehicle mass; source: https://www.tesla.com/en_au/model3
mass = 1,747  # kg

    #Vehicle yaw moment of inertia; source: https://www.studeersnel.nl/nl/document/technische-universiteit-eindhoven/road-vehicle-dynamics/exercise-1-tesla-cg/92372964
yaw_inertia = 2400  # kg*m^2, estimated

    #Vehicle wheelbase; source: https://www.tesla.com/en_au/model3
whlb = 2.875  # m
    #CoG distances using weight distribution; source: https://www.tesla.com/en_au/model3
lf = whlb * 0.47  # m, distance from CoG to front axle
lr = whlb * 0.53  # m, distance from CoG to rear

    #Stiffness coefficients; source: https://www.studeersnel.nl/nl/document/technische-universiteit-eindhoven/road-vehicle-dynamics/exercise-1-tesla-cg/92372964
Cf = 107500  # N/rad, front cornering stiffness 
Cr = 117500  # N/rad, rear cornering stiffness


#Vehicle speeds
V_x = np.array([11.0, 17.0, 22.0, 31.0])  # m/s


## Linearised bicycle model

# Slip angles and tyre forces
def slip_angles_li(y_dot, psi_dot, delta, V_i):
    theta_f = (y_dot + lf * psi_dot)/V_i
    theta_r = (y_dot - lr * psi_dot)/V_i

    alpha_f = delta - theta_f
    alpha_r = -theta_r

    return alpha_f, alpha_r

def tyre_forces_li(alpha_f, alpha_r):
    F_yf = 2 * Cf * alpha_f
    F_yr = 2 * Cr * alpha_r

    return F_yf, F_yr


# Nonlinear bicycle model
def slip_angles_nl(y_dot, psi_dot, delta, V_i):
    theta_f = np.arctan((y_dot + lf * psi_dot)/V_i)
    theta_r = np.arctan((y_dot - lr * psi_dot)/V_i)

    alpha_f = delta - theta_f
    alpha_r = -theta_r

    return alpha_f, alpha_r

def tyre_forces_nl(alpha_f, alpha_r):
    F_yf = 2 * Cf * np.tan(alpha_f)
    F_yr = 2 * Cr * np.tan(alpha_r)

    return F_yf, F_yr

# Linear state-space model: Linearised about small slip angles and small steering angle
def state_space_matrices(V_x):
    A = np.array([
        [0, 1, 0, 0],
        [0, -(2*Cf + 2*Cr)/(mass*V_x), 0, -V_x - (2*Cf*lf - 2*Cr*lr)/(mass*V_x)],
        [0, 0, 0, 1],
        [0, -(2*Cf*lf - 2*Cr*lr)/(yaw_inertia*V_x), 0, -(2*Cf*lf**2 + 2*Cr*lr**2)/(yaw_inertia*V_x)]
    ])

    B = np.array([
        [0],
        [2 * Cf / mass],
        [0],
        [2 * Cf * lf / yaw_inertia]
    ])

    return A, B