import math
import numpy as np
import matplotlib.pyplot as plt

def simpleModel(torque, wob):
    # Constants
    diameterInches = 4 # Typical width of flat bit per https://www.ditchwitch.com/sites/default/files/HDD-Tooling-Catalog.pdf
    MSE = 350 # psi - need to verify whether this is a viable number
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
torque_array_wob50 = np.array([])
wob_array_t50 = np.array([])
torque_array_wob100 = np.array([])
wob_array_t100 = np.array([])
torque_array_wob150 = np.array([])
wob_array_t150 = np.array([])
torque_array_wob200 = np.array([])
wob_array_t200 = np.array([])
range_array = np.array([])

# Fill each array by running the model at specific intervals of Torque and WOB
for i in range(0,250):
    torque_array_wob50 = np.append(torque_array_wob50, simpleModel(i, 50))
    wob_array_t50 = np.append(wob_array_t50, simpleModel(50, i))
    torque_array_wob100 = np.append(torque_array_wob100, simpleModel(i, 100))
    wob_array_t100 = np.append(wob_array_t100, simpleModel(100, i))
    torque_array_wob150 = np.append(torque_array_wob150, simpleModel(i, 150))
    wob_array_t150 = np.append(wob_array_t150, simpleModel(150, i))
    torque_array_wob200 = np.append(torque_array_wob200, simpleModel(i, 200))
    wob_array_t200 = np.append(wob_array_t200, simpleModel(200, i))

    range_array = np.append(range_array, i)


# TORQUE PLOT - plot deltaTorque based on fixed WOB values
plt.plot(range_array, torque_array_wob50, label ="WOB = 50 lbs")
plt.plot(range_array, torque_array_wob100, label ="WOB = 100 lbs")
plt.plot(range_array, torque_array_wob150, label ="WOB = 150 lbs")
plt.plot(range_array, torque_array_wob200, label ="WOB = 200 lbs")
plt.legend()
plt.title(r"$\Delta$Torque vs ROP (fixed WOB)")
plt.xlabel('Torque (ft lbs)')
plt.ylabel('ROP (in/min)')
plt.show()

# WOB Plot - plot deltaWOB based on fixed Torque values
plt.plot(range_array, wob_array_t50, label ="T = 50 ft-lbs")
plt.plot(range_array, wob_array_t100, label ="T = 100 ft-lbs")
plt.plot(range_array, wob_array_t150, label ="T = 150 ft-lbs")
plt.plot(range_array, wob_array_t200, label ="T = 200 ft-lbs")
plt.legend(loc = 'upper right')
plt.title(r"$\Delta$WOB vs ROP (fixed Torque)")
plt.xlabel('WOB (lbs)')
plt.ylabel('ROP (in/min)')
plt.show()

# UNITS
# wob = Weight on bit (lb)
# RPM = rotation per minute
# torque = Rotational Torque (in-lb)
# area = Cross section area of bit (in^2)
# ROP = rate of penetration (in/min)
