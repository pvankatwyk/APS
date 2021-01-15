# averageData.py
# Takes the current data and creates a new dataset based on averages of X number of rows. For example, makes a dataset
# combining every three rows, and subsets based on quartile values of ROP. Analyzes the potential impact of keeping the
# parameters within the quartile range.

# Change "num_rods_avg" to determine how many rods you want to average at a time.
num_average = 3

# Result:
# By keeping the drilling parameters within the first quartile ROP, we INCREASE the average speed by 35.38%.
# By keeping the drilling parameters within the second quartile ROP, we INCREASE the average speed by 8.2%.
# By keeping the drilling parameters within the third quartile ROP, we DECREASE the average speed by 14.85%.
# By keeping the drilling parameters within the fourth quartile ROP, we DECREASE the average speed by 59.67%.

import pandas as pd
import numpy as np
import math

# Import data
from Subfunctions import PrepareData
data = PrepareData(folder=r'C:/Users/Peter/Documents/Work/APS/data/', profiling=False)

# subset outliers/extreme values for rpm and torque (lots of zero values, uncomment profiling above to view)
data = data.loc[(data['Rotation Speed Max (rpm)'] > 0) & (data['Rotation Torque Max (ft-lb)'] > 0)]

# rename columns for ease of referencing
data.columns = ['rod', 'rpm', 'torque', 'thrust', 'mud_flow', 'mud_press', 'thrust_speed', 'pull_force', 'pull_speed',
                'ds_length', 'rop', 'deltaTime', 'drill', 'timestamp', 'rophr']

# drop columns that aren't needed here
data = data.drop(columns=['drill', 'timestamp', 'rophr'])

# set up parameters for average calculation
average_array = np.zeros(len(data))
sequence = range(0, len(data), num_average)

# For each column, average 3 rows at a time and add the array of averages to a dictionary
out_dict = {}
for col in data.columns:
    column_array = np.array(data[col])
    out_list = []
    for i in sequence:
        temp = 0
        if i < math.ceil(len(data) / num_average) - num_average:
            for j in range(num_average):
                temp += (column_array[i + j]) / num_average
            out_list.append(temp)
        else:
            for j in range(num_average):
                temp += (column_array[i - j]) / num_average
            out_list.append(temp)
    out_dict[col] = out_list

# Make the output dictionary into a dataframe
average_data = pd.DataFrame(out_dict)

# Run the quartile analysis
from Subfunctions import Quartile_Analysis
Quartile_Analysis(average_data)

print('Done')
