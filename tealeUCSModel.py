import math
import numpy as np

def tealeUCSModel(thrust, bit_diameter, RPM, UCS):
    friction = 0.2
    eff = 1
    bit_area = math.pi * ((bit_diameter / 2) ** 2) # A = pi*r^2
    ROP = ((math.pi/30)*RPM*(friction*bit_area/3)*(thrust/bit_area))/((UCS/eff)-(thrust/bit_area))
    return ROP