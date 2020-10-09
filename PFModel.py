import math

# OVERVIEW
# This model uses a model from Pessier & Fear (1992) which was derived from Teale (1964). Their model takes the MSE
# equation from Teale and relates Torque as a function of RPM, WOB, and friction. Belayneh (2019) then took Pessier &
# Fear's model and accounted for both vertical and rotational friction, giving us this model.

# Advantages: This model is similar to the ZTModel.py except it takes into account the friction more fully.
# Disadvantages: Doesn't rely on Zacny, 2007 auger RPM equation, but also doesn't have variable torque input because it
# is directly related to WOB and RPM. Also doesn't account for as many parameters as BYM.

def PFModel(RPM, thrust, bit_diameter, friction):
    # Constants
    MSE = 1000 # Energy needed to remove 1 volume unit of soil (NEED TO VERIFY WITH DATA)
    #bit_diameter = 4 # Typical width of flat bit per https://www.ditchwitch.com/sites/default/files/HDD-Tooling-Catalog.pdf
    #Mu = 0.6 # Coefficient of sliding friction
    Mu = friction
    borehole_area = math.pi * (bit_diameter / 2) ** 2 # Borehole area

    # Original Pessier & Fear Model
    # torque = Mu*bit_diameter*WOB / 36 -- substitute that into Teale's equation:
    #ROP = (13.33*borehole_area*RPM*Mu)/(bit_diameter*(MSE*borehole_area-wob))

    # Belayneh (2019) - Accounts for rotation friction
    inside_sqrt = 1 + ((16 * (Mu**2) * (thrust ** 2)) / (9 * (MSE * borehole_area - thrust) ** 2))
    ROP = ((5 * math.sqrt(2) * math.pi * bit_diameter * RPM) / (2)) * (math.sqrt(math.sqrt(inside_sqrt) - 1))
    return ROP