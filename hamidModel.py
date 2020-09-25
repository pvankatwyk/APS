# Model based on Hamid's analysis on slide 34 of "Automation of HDD Performance Optimization" Slides

import math
import pandas as pd

def hamidModel(diameter, helix_angle, uSurface, uWall):
    g = 9.8
    RPM_1 = (30 / math.pi) * math.sqrt(0.1 * g * ((math.tan(20 * (math.pi / 180)) + uSurface) / (diameter * uWall)))
    RPM_2 = (30 / math.pi) * math.sqrt(0.1 * g * ((math.tan(20 * (math.pi / 180)) * uSurface +1) / (diameter * uWall)))
    Torque = 85000 * (RPM_1 ** -2.3) + 1800
    Power = RPM_2 * (math.pi / 30) * Torque

    list_out = []
    for output in (diameter, helix_angle, uSurface, uWall, RPM_1, RPM_2, Power, Torque):
        list_out.append(output)

    return (list_out)

Results_DF = pd.DataFrame(columns = ["Diameter", "Helix Angle", "uSurface", "uWall", "RPM_1", "RPM_2", "Torque", "Power"])

for diameter in (0.08, 0.16):
    for helix_angle in (20,25):
        for uSurface in (0.2,0.3):
            for uWall in (0.5,0.6,0.7,0.8):
                Results_DF.loc[len(Results_DF)] = hamidModel(diameter,helix_angle,uSurface,uWall)

print(Results_DF)