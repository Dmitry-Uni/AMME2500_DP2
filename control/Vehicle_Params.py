import numpy as np
### Vehicle Parameters - Tesla Model 3

g = 9.81  # m/s^2, gravitational acceleration

    #Vehicle mass; source: https://www.tesla.com/en_au/model3
mass = 1747  # kg

    #Vehicle yaw moment of inertia; source: https://www.studeersnel.nl/nl/document/technische-universiteit-eindhoven/road-vehicle-dynamics/exercise-1-tesla-cg/92372964
yaw_inertia = 2400  # kg*m^2, estimated

    #Vehicle wheelbase; source: https://www.tesla.com/en_au/model3
whlb = 2.875  # m
    #CoG distances using weight distribution; source: https://www.tesla.com/en_au/model3
lr = whlb * 0.47  # m, distance from CoG to rear axle
lf = whlb * 0.53  # m, distance from CoG to front axle

    #Stiffness coefficients; source: https://www.studeersnel.nl/nl/document/technische-universiteit-eindhoven/road-vehicle-dynamics/exercise-1-tesla-cg/92372964
Cf_tire = 107500  # N/rad, front tire cornering stiffness 
Cr_tire = 117500  # N/rad, rear tire cornering stiffness
Cf = Cf_tire * 2  # N/rad, total front cornering stiffness (2 tires)
Cr = Cr_tire * 2  # N/rad, total rear cornering stiffness

    #Tire friction coefficient; source: https://www.tyrereviews.com/Tyre/Michelin/e.Primacy.htm
tire_friction_coefficient_cases = [0.9, 0.6, 0.4, 0.1] # Dry, wet, sandy road, oil slick conditions
mu = tire_friction_coefficient_cases[0]  # Using dry road condition for control design

    #Steering angle sensor sampling rate; source: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/itr2.12085
steering_sampling_rate = 100  # Hz

    #Maximum steering angle; source: https://www.tesla.com/en_au/model3
max_steering_angle = np.radians(36)  # radians, typical for passenger

    #Vehicle speeds
V_i = np.array([11.0, 17.0, 22.0, 31.0])  # m/s
V_x = V_i[1]  # m/s, initial speed for control design

    # Initial state of the vehicle (lateral error, heading error, lateral velocity, yaw rate)
initial_state = np.array([[0.0], [0.0], [0.0], [0.0]])  # [e_y, e_psi, v_y, r]
initial_position = [5, 2.6]  # (x, y) 