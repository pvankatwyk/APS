# Model based on Hamid's analysis on slide 34 of "Automation of HDD Performance Optimization" Slides

# RPM = 0.1*(30/pi)*sqrt(((2*g)/Do)*((MuS*tanTheta+1)/MuW))
# Power = RPM * (2pi/60) * Torque
# Torque = 85000 * RPM^-2.3 + 1800

import math

def hamidModel(diameter, helix_angle, uSurface, uWall):
    g = 9.81 # Gravity

    RPM = (30 / math.pi) * math.sqrt(((2 * g * (math.tan(helix_angle * (math.pi / 180)) + uSurface)) / (diameter * uWall)))
    Torque = 85000 * (RPM**-2.7) + 1800
    Power = RPM * ((2*math.pi)/60)*Torque
    out = 'RPM: ' + str(round(RPM,2)) + ', Torque: ' + str(round(Torque,2)) + ', Power: ' + str(round(Power,2))
    return print(out)

hamidModel(0.8,20,.2,.5)