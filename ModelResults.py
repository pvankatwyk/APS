# This script runs all of the models and graphs the results.

import numpy as np
import matplotlib.pyplot as plt
from ZTModel import ZTModel
from PFModel import PFModel
from BYModel import BYModel

# ZT Model --------------------------------------------------------

# Preallocate arrays for graphs
torque_array_thrust500 = np.array([])
thrust_array_t500 = np.array([])
torque_array_thrust1000 = np.array([])
thrust_array_t1000 = np.array([])
torque_array_thrust1500 = np.array([])
thrust_array_t1500 = np.array([])
torque_array_thrust2000 = np.array([])
thrust_array_t2000 = np.array([])
range_array = np.array([])
bit_diameter = 4
friction = 0.6

# Fill each array by running the model at specific intervals of Torque and thrust
for i in range(0,4000):
    torque_array_thrust500 = np.append(torque_array_thrust500, ZTModel(i, 500, bit_diameter, friction))
    thrust_array_t500 = np.append(thrust_array_t500, ZTModel(500, i, bit_diameter, friction))
    torque_array_thrust1000 = np.append(torque_array_thrust1000, ZTModel(i, 1000, bit_diameter, friction))
    thrust_array_t1000 = np.append(thrust_array_t1000, ZTModel(1000, i, bit_diameter, friction))
    torque_array_thrust1500 = np.append(torque_array_thrust1500, ZTModel(i, 1500, bit_diameter, friction))
    thrust_array_t1500 = np.append(thrust_array_t1500, ZTModel(1500, i, bit_diameter, friction))
    torque_array_thrust2000 = np.append(torque_array_thrust2000, ZTModel(i, 2000, bit_diameter, friction))
    thrust_array_t2000 = np.append(thrust_array_t2000, ZTModel(2000, i, bit_diameter, friction))

    range_array = np.append(range_array, i)


# TORQUE PLOT - plot deltaTorque based on fixed thrust values
plt.plot(range_array, torque_array_thrust500, label ="Thrust = 500 lbs")
plt.plot(range_array, torque_array_thrust1000, label ="Thrust = 1000 lbs")
plt.plot(range_array, torque_array_thrust1500, label ="Thrust = 1500 lbs")
plt.plot(range_array, torque_array_thrust2000, label ="Thrust = 2000 lbs")
plt.legend()
plt.title(r"ZT Model $\Delta$Torque vs ROP (fixed Thrust)")
plt.xlabel('Torque (ft lbs)')
plt.ylabel('ROP (in/min)')
plt.show()

# thrust Plot - plot deltathrust based on fixed Torque values
plt.plot(range_array, thrust_array_t500, label ="T = 500 ft-lbs")
plt.plot(range_array, thrust_array_t1000, label ="T = 1000 ft-lbs")
plt.plot(range_array, thrust_array_t1500, label ="T = 1500 ft-lbs")
plt.plot(range_array, thrust_array_t2000, label ="T = 2000 ft-lbs")
plt.legend(loc = 'upper right')
plt.title(r"ZT Model $\Delta$Thrust vs ROP (fixed Torque)")
plt.xlabel('Thrust (lbs)')
plt.ylabel('ROP (in/min)')
plt.show()

# UNITS
# thrust = Weight on bit (lb)
# RPM = rotation per minute
# torque = Rotational Torque (in-lb)
# area = Cross section area of bit (in^2)
# ROP = rate of penetration (in/min)


# PFModel -------------------------------------------------------

# Preallocate arrays for graphs
RPM_array_thrust500 = np.array([])
thrust_array_r50 = np.array([])
RPM_array_thrust1000 = np.array([])
thrust_array_r100 = np.array([])
RPM_array_thrust1500 = np.array([])
thrust_array_r150 = np.array([])
RPM_array_thrust2000 = np.array([])
thrust_array_r200 = np.array([])
range_array_long = np.array([])
range_array_short = np.array([])

# Fill each array by running the model at specific intervals of Torque and thrust
for i in range(0,4000):
    thrust_array_r50 = np.append(thrust_array_r50, PFModel(50, i, bit_diameter, friction))
    thrust_array_r100 = np.append(thrust_array_r100, PFModel(100, i, bit_diameter, friction))
    thrust_array_r150 = np.append(thrust_array_r150, PFModel(150, i, bit_diameter, friction))
    thrust_array_r200 = np.append(thrust_array_r200, PFModel(200, i, bit_diameter, friction))
    range_array_long = np.append(range_array_long, i)

# Arrays with RPM Values at given thrust
for j in range(0, 500):
    RPM_array_thrust500 = np.append(RPM_array_thrust500, PFModel(j, 500, bit_diameter, friction))
    RPM_array_thrust1000 = np.append(RPM_array_thrust1000, PFModel(j, 1000, bit_diameter, friction))
    RPM_array_thrust1500 = np.append(RPM_array_thrust1500, PFModel(j, 1500, bit_diameter, friction))
    RPM_array_thrust2000 = np.append(RPM_array_thrust2000, PFModel(j, 2000, bit_diameter, friction))
    range_array_short = np.append(range_array_short, j)


# TORQUE PLOT - plot deltaTorque based on fixed thrust values
plt.plot(range_array_short, RPM_array_thrust500, label ="Thrust = 500 lbs")
plt.plot(range_array_short, RPM_array_thrust1000, label ="Thrust = 1000 lbs")
plt.plot(range_array_short, RPM_array_thrust1500, label ="Thrust = 1500 lbs")
plt.plot(range_array_short, RPM_array_thrust2000, label ="Thrust = 2000 lbs")
plt.legend()
plt.title(r"PF Model $\Delta$RPM vs ROP (fixed Thrust)")
plt.xlabel('RPM (rev/min)')
plt.ylabel('ROP')
plt.show()

# thrust Plot - plot deltathrust based on fixed Torque values
plt.plot(range_array_long, thrust_array_r50, label ="RPM = 50 rev/min")
plt.plot(range_array_long, thrust_array_r100, label ="RPM = 100 rev/min")
plt.plot(range_array_long, thrust_array_r150, label ="RPM = 150 rev/min")
plt.plot(range_array_long, thrust_array_r200, label = "RPM = 200 rev/min")
plt.legend(loc = 'upper right')
plt.title(r"PF Model $\Delta$Thrust vs ROP (fixed RPM)")
plt.xlabel('Thrust (lbs)')
plt.ylabel('ROP')
plt.show()

# UNITS
# thrust = Weight on bit (lb)
# RPM = rotation per minute
# torque = Rotational Torque (in-lb)
# area = Cross section area of bit (in^2)
# ROP = rate of penetration (ft/hr)

