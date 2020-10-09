
def BYModel():
    import math
    # Model Coefficients
    # Comments - Suggested ranges for constants by Bourgoyne and Young
    a1 = 3.91 # 0.5 < a1 < 1.9 - Formation Strength
    a2 = 0.0000945 # 0.000001 < a2 < 0.0005 - Normal Compaction
    a3 = 0.0000686 # 0.000001 < a3 < 0.0009 - Under Compaction
    a4 = 0.0000864 # 0.000001 < a4 < 0.0001 - Pressure Differential
    a5 = 0.37 # 0.5 < a5 < 2 - Weight on bit
    a6 = 2.23 # 0.5 < a6 < 1 - Rotary Speed
    a7 = 0.025 # 0.3 < a7 < 1.5 - Tooth Wear
    a8 = 0.67 # 0.3 < a8 < 0.6 - Jet Impact Force

    # Papers that may be of use to determining coefficients --
        # Determination of constant coefficients of Bourgoyne and Young drilling
            # rate model using a novel evolutionary algorithm
        # Mathematical Modeling Applied to Drilling
            # Engineering: An Application of Bourgoyne and Young ROP
            # Model to a Presalt Case Study
        # ROP Modeling for Volcanic Geothermal Drilling Optimization

    # Model Constants
    depth = 2150 # True vertical depth (feet)
    pore_pressure = 8.365 # Pore pressure gradient (lb(m)/gallon)
    WOB = .82 # Weight on bit (1000 lb(f))
    diameter = 4 # Bit diameter (in)
    threshold_weight = 1000 # "threshold bit weight per inch of bit diameter"
    RPM = 120 # Rotations per Minute
    tooth_wear = -0.5 # Fractional bit tooth wear
    Fj = 0.882 # # Jet Impact Force (lb(f))
    mud_density = 8.93 # Equivalent mud density (lb(m)/gal)

    # ROP Factors
    f1 = math.exp(2.303*a1)
    f2 = math.exp(2.303*a2*(10000-depth))
    f3 = math.exp(2.303 * a3 * (depth**0.69) * (pore_pressure - 9))
    f4 = math.exp(2.303 * a4 * depth * (pore_pressure - mud_density))
    f5 = (((WOB/diameter) - threshold_weight) / (4 - threshold_weight)) ** a5
    f6 = (RPM/100)**a6
    f7 = math.exp(-a7*tooth_wear)
    f8 = (Fj/1000)**a8

    # Final Equation
    ROP = f1 * f2 * f3 * f4 * f5 * f6 * f7 * f8
    return ROP


print(BYModel())