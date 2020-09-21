import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def modelTorque(y,t):
    RPM = 100 # Value calculated in main.py
    MSE = 350
    W = 100
    diameterInches = 4
    bit_area = math.pi * ((diameterInches / 2) ** 2) # A = pi*r^2
    dROPdT = (2*math.pi*RPM)/((bit_area*MSE)-W)
    return dROPdT

def modelWOB(W,t):
    RPM = 100
    MSE = 350
    T = 100
    diameterInches = 4
    bit_area = math.pi * ((diameterInches / 2) ** 2) # A = pi*r^2
    dROPdW = (2*math.pi*RPM*T)/(((MSE*bit_area)-W)**2)
    return dROPdW


y0 = 0
t = np.linspace(0,200)
y = odeint(modelTorque, y0, t)
z = odeint(modelWOB, y0, t)
plt.plot(t,y)
plt.plot(t,z)
plt.xlabel('Torque (ft lbs) / Weight on Bit (lbs)')
plt.ylabel('ROP (in/min)')
plt.title(r"ODEInt Model of $\Delta$Torque and $\Delta$WOB vs ROP")
plt.legend()
plt.show()



# wob = Weight on bit (lb)
# RPM = rotation per minute
# torque = Rotational Torque (in-lb)
# area = Cross section area of bit (in^2)
# ROP = rate of penetration (in/min)
