# makeDataset:
#   - Inputs: (Integers) Lower and Upper bounds for drilling parameters
#   - Output: (DataFrame) Synthetic dataset with randomly generated numbers between the lower and upper bounds of
#             each parameter as inputted by the user. Used to train ML models.
from PFModel import PFModel


def makeDataset(dataset_length, RPM_lower, RPM_upper, torque_lower, torque_upper, thrust_lower, thrust_upper, bit_lower,
                bit_upper, friction_lower, friction_upper):
    from ZTModel import ZTModel
    import random
    import pandas as pd
    import numpy as np

    RPM = np.random.randint(RPM_lower, RPM_upper + 1, dataset_length)
    torque = np.random.randint(torque_lower, torque_upper + 1, dataset_length)
    thrust = np.random.randint(thrust_lower, thrust_upper + 1, dataset_length)
    bit_diameter = np.random.randint(bit_lower, bit_upper + 1, dataset_length)
    friction_coeff = []
    for i in range(dataset_length):
        friction_coeff.append(friction_lower + (friction_upper - friction_lower) * random.random())

    dataset = pd.DataFrame()
    dataset['thrust'] = thrust
    dataset['torque'] = torque
    dataset['RPM'] = RPM
    dataset['bit_diameter'] = bit_diameter
    dataset['friction_coeff'] = friction_coeff

    dataset['classification'] = dataset['friction_coeff'].apply(lambda x: soilClassifier(x))

    dataset['ZT_ROP'] = dataset.apply(lambda x: ZTModel(x.torque, x.thrust, x.bit_diameter, x.friction_coeff),
                                         axis=1)
    dataset['PF_ROP'] = dataset.apply(lambda x: PFModel(x.RPM, x.thrust, x.bit_diameter, x.friction_coeff), axis=1)
    return dataset

# soilClassifier:
#   - Input: (Integer) Friction coefficient of the soil
#   - Output: (String) Soil Classification based on the inputted friction coefficient
def soilClassifier(x):
    if x < 0.47:
        out = 'Organic Silt or Clay'
    elif x > 0.47 and x <= 0.59:
        out = 'Clay'
    elif x > 0.59 and x <= 0.63:
        out = 'Clayey Silt or Sand'
    elif x > 0.63 and x <= 0.65:
        out = 'Silt'
    elif x > 0.65 and x <= 0.69:
        out = 'Silty Sand'
    elif x > 0.69 and x <= 0.75:
        out = 'Silty Gravel'
    elif x > 0.75 and x <= 0.80:
        out = 'Fine to Course Sand'
    elif x > 0.8:
        out = 'Fine to Course Gravel'
    return out