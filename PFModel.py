import math
import numpy as np
import matplotlib.pyplot as plt
def PFModel(RPM, wob):
    # Constants

    MSE = 350 # Energy needed to remove 1 volume unit of soil (NEED TO VERIFY WITH DATA)
    bit_size = 4 # Typical width of flat bit per https://www.ditchwitch.com/sites/default/files/HDD-Tooling-Catalog.pdf
    Mu = 0.6 # Coefficient of sliding friction
    borehole_area = math.pi*(bit_size/2)**2 # Borehole area
    # torque = Mu*bit_size*WOB / 36 -- substitute that into Teale's equation:
    #ROP = (13.33*borehole_area*RPM*Mu)/(bit_size*(MSE*borehole_area-wob))
    inside_sqrt = 1 + ((16*(Mu**2)*(wob**2))/(9*(MSE*borehole_area-wob)**2))
    ROP = ((5*math.sqrt(2)*math.pi*bit_size*RPM)/(2))*(math.sqrt(math.sqrt(inside_sqrt)-1))
    return ROP


# Preallocate arrays for graphs
RPM_array_wob500 = np.array([])
wob_array_r50 = np.array([])
RPM_array_wob1000 = np.array([])
wob_array_r100 = np.array([])
RPM_array_wob1500 = np.array([])
wob_array_r150 = np.array([])
RPM_array_wob2000 = np.array([])
wob_array_r200 = np.array([])
range_array_long = np.array([])
range_array_short = np.array([])

# Fill each array by running the model at specific intervals of Torque and WOB
for i in range(0,4000):
    wob_array_r50 = np.append(wob_array_r50, PFModel(50, i))
    wob_array_r100 = np.append(wob_array_r100, PFModel(100, i))
    wob_array_r150 = np.append(wob_array_r150, PFModel(150, i))
    wob_array_r200 = np.append(wob_array_r200, PFModel(200, i))
    range_array_long = np.append(range_array_long, i)

# Arrays with RPM Values at given WOB
for j in range(0, 500):
    RPM_array_wob500 = np.append(RPM_array_wob500, PFModel(j, 500))
    RPM_array_wob1000 = np.append(RPM_array_wob1000, PFModel(j, 1000))
    RPM_array_wob1500 = np.append(RPM_array_wob1500, PFModel(j, 1500))
    RPM_array_wob2000 = np.append(RPM_array_wob2000, PFModel(j, 2000))
    range_array_short = np.append(range_array_short, j)


# TORQUE PLOT - plot deltaTorque based on fixed WOB values
plt.plot(range_array_short, RPM_array_wob500, label ="Thrust = 500 lbs")
plt.plot(range_array_short, RPM_array_wob1000, label ="Thrust = 1000 lbs")
plt.plot(range_array_short, RPM_array_wob1500, label ="Thrust = 1500 lbs")
plt.plot(range_array_short, RPM_array_wob2000, label ="Thrust = 2000 lbs")
plt.legend()
plt.title(r"$\Delta$RPM vs ROP (fixed Thrust)")
plt.xlabel('RPM (rev/min)')
plt.ylabel('ROP (in/min)')
plt.show()

# WOB Plot - plot deltaWOB based on fixed Torque values
plt.plot(range_array_long, wob_array_r50, label ="RPM = 50 rev/min")
plt.plot(range_array_long, wob_array_r100, label ="RPM = 100 rev/min")
plt.plot(range_array_long, wob_array_r150, label ="RPM = 150 rev/min")
plt.plot(range_array_long, wob_array_r200, label = "RPM = 200 rev/min")
plt.legend(loc = 'upper right')
plt.title(r"$\Delta$Thrust vs ROP (fixed RPM)")
plt.xlabel('Thrust (lbs)')
plt.ylabel('ROP (in/min)')
plt.show()

# UNITS
# wob = Weight on bit (lb)
# RPM = rotation per minute
# torque = Rotational Torque (in-lb)
# area = Cross section area of bit (in^2)
# ROP = rate of penetration (ft/hr)