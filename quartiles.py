# quartiles.py
# Takes the current data and subsets based on quartile values of ROP. Analyzes the potential impact of keeping the
# parameters within the quartile range.

# An option is to analyze the average ROP over X rods at a time. This increases resolution on the ROP measurement and
# decreases each rods accuracy. Change this variable "average" to False if you want to run the analysis on the raw data.
# Change "num_rods_avg" to determine how many rods you want to average at a time (recommended 5).
average = False
num_rods_avg = 5

# Result:
# By keeping the drilling parameters within the first quartile ROP, we increase the average speed by 15.01%.
# By keeping the drilling parameters within the second quartile ROP, we increase the average speed by 7.12%.
# By keeping the drilling parameters within the third quartile ROP, we decrease the average speed by 46.55%.
# By keeping the drilling parameters within the fourth quartile ROP, we decrease the average speed by 82.68%.

from Subfunctions import PrepareData
import numpy as np

# Prepare data based on Hedegren's "visualize.py"
data = PrepareData(folder=r'C:/Users/Peter/Documents/Work/APS/data/', profiling=False)

# Output Pands Profiling for EDA
# file = r'C:/Users/Peter/Downloads/CleanData.html'
# from pandas_profiling import ProfileReport
# prof = ProfileReport(data)
# prof.to_file(output_file=file)

# subset outliers/extreme values for rpm and torque (lots of zero values, uncomment profiling above to view)
data = data.loc[(data['Rotation Speed Max (rpm)'] > 0) & (data['Rotation Torque Max (ft-lb)'] > 0)]

# rename columns for ease of referencing
data.columns = ['rod', 'rpm', 'torque', 'thrust', 'mud_flow', 'mud_press', 'thrust_speed', 'pull_force', 'pull_speed',
                'ds_length', 'rop', 'deltaTime', 'drill', 'timestamp', 'rophr']

# If you want to change the ROP to use average values over X number of rods (see script description)...
if average:
    seconds_array = np.array(data.deltaTime)
    avg_val = np.zeros(len(data))  # Preallocate
    sequence = range(0, len(data), num_rods_avg)  # Create sequence of indexes to be looped through
    # For each index, calculate the average of the following X rods and add them up
    for i in sequence:
        for j in range(num_rods_avg):
            if i < len(data) - num_rods_avg:
                avg_val[i] += seconds_array[i + j] / num_rods_avg
            else:
                avg_val[i] += seconds_array[i - j] / num_rods_avg  # The last calculations look back to average
    # Fill the zero values with their corresponding averages
    # example: [258,0,0,0,,0,84,0,0,0...] to [258,258,258,258,258,84,84,84...]
    for i in range(len(data)):
        if avg_val[i] == 0:
            avg_val[i] = avg_val[i - 1]
        else:
            pass
    # Calculate new ROP based on average time calculated and place into dataset
    rop_avg = 60.0 * 10.0 / avg_val
    del data['rop']
    data["rop"] = rop_avg
    data = data[['rod', 'rpm', 'torque', 'thrust', 'mud_flow', 'mud_press', 'thrust_speed', 'pull_force', 'pull_speed',
                 'ds_length', 'rop', 'deltaTime', 'drill', 'timestamp', 'rophr']]

from Subfunctions import Quartile_Analysis
Quartile_Analysis(data)

print('Done')