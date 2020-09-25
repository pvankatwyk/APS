import math
def BourgoyneYoung():
    # Model Coefficients
    # Comments - Suggested ranges for constants by Bourgoyne and Young
    a1 = 0 # 0.5 < a1 < 1.9
    a2 = 0 # 0.000001 < a2 < 0.0005
    a3 = 0 # 0.000001 < a3 < 0.0009
    a4 = 0 # 0.000001 < a4 < 0.0001
    a5 = 0 # 0.5 < a5 < 2
    a6 = 0 # 0.5 < a6 < 1
    a7 = 0 # 0.3 < a7 < 1.5
    a8 = 0 # 0.3 < a8 < 0.6

    # Papers that may be of use to determining coefficients --
        # Determination of constant coefficients of Bourgoyne and Young drilling
            # rate model using a novel evolutionary algorithm
        # Mathematical Modeling Applied to Drilling
            # Engineering: An Application of Bourgoyne and Young ROP
            # Model to a Presalt Case Study
        # ROP Modeling for Volcanic Geothermal Drilling Optimization

    # Model Constants
    depth = 0 # True vertical depth (feet)
    pore_pressure = 0 # Pore pressure gradient (lb(m)/gallon)
    WOB = 0 # Weight on bit (1000 lb(f))
    diameter = 0 # Bit diameter (in)
    threshold_weight = 0 # "threshold bit weight per inch of bit diameter"
    RPM = 0 # Rotations per Minute
    tooth_wear = 0 # Fractional bit tooth wear
    Fj = 0 # # Jet Impact Force (lb(f))
    mud_density = 0 # Equivalent mud density (lb(m)/gal)

    # ROP Factors
    f1 = math.exp(2.303*a1)
    f2 = math.exp(2.303*a2*(10000-depth))
    f3 = math.exp(2.303 * a3 * (depth**0.69) * (pore_pressure - mud_density))
    f4 = math.exp(2.303 * a4 * depth * (pore_pressure - mud_density))
    f5 = (((WOB/diameter) - threshold_weight) / (4 - threshold_weight)) ** a5
    f6 = (RPM/60)**a6
    f7 = math.exp(-a7*tooth_wear)
    f8 = (Fj/1000)**a8

    # Final Equation
    ROP = f1 * f2 * f3 * f4 * f5 * f6 * f7 * f8
    return ROP