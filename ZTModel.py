import math


# OVERVIEW
# ZT Model = Zacney & Teale Model
# This model takes inputs into Zacny, 2007 RPM equation, which calculates the minimum required RPM for a auger-type bit
# to remove soil. Then, that RPM and inputted RPM and Torque are used in Teale's MSE equation (which is solved for ROP),
# to calculate ROP.

# Advantages: Takes into account bit dimensions, friction, and other parameters we see as necessary for the model.
# Disadvantages: RPM is calculated as the minimum RPM required for an AUGER to remove soil. This may not apply to
#   different bit types such as spoon or trihawk. Because of this, the friction is calculated using friction of the
#   blades on the soil as well as the soil on the scroll, which seemingly doesn't apply outside of an auger. This model
#   also does not account for pressure, mud hydraulics, etc.

def ZTModel(torque, thrust, bit_diameter, friction):

    # Constants
    #bit_diameter = 4 # Typical width of flat bit per https://www.ditchwitch.com/sites/default/files/HDD-Tooling-Catalog.pdf
    MSE = 2500 # psi - NEED TO GET AN ACCURATE NUMBER FOR THIS (USE TEALE'S USING EXPECTED VALUES)
    bit_area = math.pi * ((bit_diameter / 2) ** 2) # A = pi*r^2
    g = 9.81 # Gravity

    # Calculate minimum RPM (Zacny 2007)
    helix_angle = 10 # Pitch angle, or blade angle, as seen on ditchwitch link above
    diameterM = bit_diameter * 0.0254 # Diameter of bit in meters (to calculate RPM)
    #uSurface = 0.25 # Friction coefficient of the scroll against soil (middle pipe) -- for auger, not sure I need here?
    #uWall = 0.75 # Friction coefficient of blade against soil -- Zacny, 2007
    # RPM Equation - Minimum RPM required to remove soil
    RPM = (30 / math.pi) * math.sqrt(((2 * g * (math.tan(helix_angle * (math.pi / 180)))) / (diameterM * friction)))

    # Calculate ROP based on MSE Equation given be Teale, 1964
    ROP = (2 * math.pi * RPM * torque) / ((bit_area * MSE) - thrust)
    #ROP = (ROP/12)*60
    #out = "WOB: "+str(round(wob,2))+ ", T: "+str(round(torque,2)) + ", RPM: "+str(round(RPM,2))+", ROP: " + str(round(ROP, 2)) + " in/min or " + str(round((ROP / 12)*60, 2)) + " ft/hr."
    return ROP