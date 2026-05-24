import numpy as np
### Vehicle Parameters - Tesla Model 3

    #Vehicle mass; source: https://www.tesla.com/en_au/model3
mass = 1747  # kg

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

    #Steering angle sensor sampling rate; source: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/itr2.12085
steering_sampling_rate = 100  # Hz

    #Maximum steering angle; source: https://www.tesla.com/en_au/model3
max_steering_angle = np.radians(36)  # radians, typical for passenger

#Vehicle speeds
V_i = np.array([11.0, 17.0, 22.0, 31.0])  # m/s
V_x = V_i[0]  # m/s, initial speed for control design