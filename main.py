import math
import numpy as np
import matplotlib.pyplot as plt

def simpleModel(torque, wob):
    # Constants
    diameterInches = 4 # Typical width of flat bit per https://www.ditchwitch.com/sites/default/files/HDD-Tooling-Catalog.pdf
    MSE = 350 # psi - NEED TO GET AN ACCURATE NUMBER FOR THIS (USE TEALE'S USING EXPECTED VALUES)
    bit_area = math.pi * ((diameterInches / 2) ** 2) # A = pi*r^2
    g = 9.81 # Gravity

    # Calculate minimum RPM (Zacny 2007)
    helix_angle = 10 # Pitch angle, or blade angle, as seen on ditchwitch link above
    diameterM = diameterInches * 0.0254 # Diameter of bit in meters (to calculate RPM)
    uSurface = 0.25 # Friction coefficient of the scroll against soil (middle pipe) -- for auger, not sure I need here?
    uWall = 0.75 # Friction coefficient of blade against soil -- Zacny, 2007
    # RPM Equation - Minimum RPM required to remove soil
    RPM = (30 / math.pi) * math.sqrt(((2 * g * (math.tan(helix_angle * (math.pi / 180)) + uSurface)) / (diameterM * uWall)))

    # Calculate ROP based on MSE Equation given be Teale, 1964
    ROP = (2 * math.pi * RPM * torque) / ((bit_area * MSE) - wob)
    #out = "WOB: "+str(round(wob,2))+ ", T: "+str(round(torque,2)) + ", RPM: "+str(round(RPM,2))+", ROP: " + str(round(ROP, 2)) + " in/min or " + str(round((ROP / 12)*60, 2)) + " ft/hr."
    return ROP

# Preallocate arrays for graphs
torque_array_wob500 = np.array([])
wob_array_t500 = np.array([])
torque_array_wob1000 = np.array([])
wob_array_t1000 = np.array([])
torque_array_wob1500 = np.array([])
wob_array_t1500 = np.array([])
torque_array_wob2000 = np.array([])
wob_array_t2000 = np.array([])
range_array = np.array([])

# Fill each array by running the model at specific intervals of Torque and WOB
for i in range(0,4000):
    torque_array_wob500 = np.append(torque_array_wob500, simpleModel(i, 500))
    wob_array_t500 = np.append(wob_array_t500, simpleModel(500, i))
    torque_array_wob1000 = np.append(torque_array_wob1000, simpleModel(i, 1000))
    wob_array_t1000 = np.append(wob_array_t1000, simpleModel(1000, i))
    torque_array_wob1500 = np.append(torque_array_wob1500, simpleModel(i, 1500))
    wob_array_t1500 = np.append(wob_array_t1500, simpleModel(1500, i))
    torque_array_wob2000 = np.append(torque_array_wob2000, simpleModel(i, 2000))
    wob_array_t2000 = np.append(wob_array_t2000, simpleModel(2000, i))

    range_array = np.append(range_array, i)


# TORQUE PLOT - plot deltaTorque based on fixed WOB values
plt.plot(range_array, torque_array_wob500, label ="Thrust = 500 lbs")
plt.plot(range_array, torque_array_wob1000, label ="Thrust = 1000 lbs")
plt.plot(range_array, torque_array_wob1500, label ="Thrust = 1500 lbs")
plt.plot(range_array, torque_array_wob2000, label ="Thrust = 2000 lbs")
plt.legend()
plt.title(r"$\Delta$Torque vs ROP (fixed Thrust)")
plt.xlabel('Torque (ft lbs)')
plt.ylabel('ROP (in/min)')
plt.show()

# WOB Plot - plot deltaWOB based on fixed Torque values
plt.plot(range_array, wob_array_t500, label ="T = 500 ft-lbs")
plt.plot(range_array, wob_array_t1000, label ="T = 1000 ft-lbs")
plt.plot(range_array, wob_array_t1500, label ="T = 1500 ft-lbs")
plt.plot(range_array, wob_array_t2000, label ="T = 2000 ft-lbs")
plt.legend(loc = 'upper right')
plt.title(r"$\Delta$Thrust vs ROP (fixed Torque)")
plt.xlabel('Thrust (lbs)')
plt.ylabel('ROP (in/min)')
plt.show()

# UNITS
# wob = Weight on bit (lb)
# RPM = rotation per minute
# torque = Rotational Torque (in-lb)
# area = Cross section area of bit (in^2)
# ROP = rate of penetration (in/min)
