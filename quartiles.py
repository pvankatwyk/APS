# quartiles.py
# Takes the current data and subsets based on quartile values of ROP. Analyzes the potential impact of keeping the
# parameters within the quartile range.

# Result:
# By keeping the drilling parameters within the first quartile ROP, we increase the average speed by 15.0146%.
# By keeping the drilling parameters within the second quartile ROP, we increase the average speed by 7.12%.
# By keeping the drilling parameters within the third quartile ROP, we decrease the average speed by 46.55%.
# By keeping the drilling parameters within the fourth quartile ROP, we decrease the average speed by 82.68%.

from Subfunctions import PrepareData
import numpy as np
import pandas as pd

# Prepare data based on Hedegren's "visualize.py"
data = PrepareData(folder = r'C:/Users/Peter/Documents/Work/APS/data/', profiling = False)

# Output Pands Profiling for EDA
# file = r'C:/Users/Peter/Downloads/CleanData.html'
# from pandas_profiling import ProfileReport
# prof = ProfileReport(data)
# prof.to_file(output_file=file)

# subset outliers/extreme values for rpm and torque (lots of zero values, uncomment profiling above to view)
data = data.loc[(data['Rotation Speed Max (rpm)'] > 0) & (data['Rotation Torque Max (ft-lb)'] > 0)]

# rename columns for ease of referencing
data.columns = ['rod', 'rpm', 'torque', 'thrust', 'mud_flow', 'mud_press', 'thrust_speed', 'pull_force', 'pull_speed', 'ds_length', 'rop', 'deltaTime', 'drill', 'timestamp', 'rophr']

# Calculate quartiles boundaries
quartiles = np.array(data.rop.quantile([0.25,0.5,0.75]))
quartile1 = quartiles[2]
quartile2 = quartiles[1]
quartile3 = quartiles[0]

# Quartile 1 Range - 10 (Q1 to maximum)
# Quartile 2 Range - 8.6956 to 10 (Q2 to Q1)
# Quartile 3 Range - 3.3333 to 8.6956 (Q3 to Q2)
# Quartile 4 Range - 0.0024 to 3.3333 (minimum to Q3)

quartile1_dataset = data.loc[data.rop >= quartile1]
quartile2_dataset = data.loc[(data.rop < quartile1) & (data.rop >= quartile2)]
quartile3_dataset = data.loc[(data.rop < quartile2) & (data.rop >= quartile3)]
quartile4_dataset = data.loc[(data.rop < quartile3) & (data.rop >= min(data.rop))]

# Make "summary" dataset to view the ROP characteristics of each quartile
rpm = np.array([np.mean(quartile1_dataset.rpm), np.mean(quartile2_dataset.rpm), np.mean(quartile3_dataset.rpm), np.mean(quartile4_dataset.rpm)])
torque = np.array([np.mean(quartile1_dataset.torque), np.mean(quartile2_dataset.torque), np.mean(quartile3_dataset.torque), np.mean(quartile4_dataset.torque)])
thrust = np.array([np.mean(quartile1_dataset.thrust), np.mean(quartile2_dataset.thrust), np.mean(quartile3_dataset.thrust), np.mean(quartile4_dataset.thrust)])
mud_flow = np.array([np.mean(quartile1_dataset.mud_flow), np.mean(quartile2_dataset.mud_flow), np.mean(quartile3_dataset.mud_flow), np.mean(quartile4_dataset.mud_flow)])
mud_press = np.array([np.mean(quartile1_dataset.mud_press), np.mean(quartile2_dataset.mud_press), np.mean(quartile3_dataset.mud_press), np.mean(quartile4_dataset.mud_press)])
thrust_speed = np.array([np.mean(quartile1_dataset.thrust_speed), np.mean(quartile2_dataset.thrust_speed), np.mean(quartile3_dataset.thrust_speed), np.mean(quartile4_dataset.thrust_speed)])
pull_force = np.array([np.mean(quartile1_dataset.pull_force), np.mean(quartile2_dataset.pull_force), np.mean(quartile3_dataset.pull_force), np.mean(quartile4_dataset.pull_force)])
pull_speed = np.array([np.mean(quartile1_dataset.pull_speed), np.mean(quartile2_dataset.pull_speed), np.mean(quartile3_dataset.pull_speed), np.mean(quartile4_dataset.pull_speed)])
ds_length = np.array([np.mean(quartile1_dataset.ds_length), np.mean(quartile2_dataset.ds_length), np.mean(quartile3_dataset.ds_length), np.mean(quartile4_dataset.ds_length)])
rop = np.array([np.mean(quartile1_dataset.rop), np.mean(quartile2_dataset.rop), np.mean(quartile3_dataset.rop), np.mean(quartile4_dataset.rop)])

summary = pd.DataFrame([rpm, torque, thrust, mud_flow, mud_press, thrust_speed, pull_force, pull_speed, ds_length, rop])
summary = summary.transpose()
summary.columns = ['rpm', 'torque', 'thrust', 'mud_flow', 'mud_press', 'thrust_speed', 'pull_force', 'pull_speed', 'ds_length', 'rop']
summary.index = ['quartile_1', 'quartile_2', 'quartile_3', 'quartile_4']

# To view "summary" DF, either view in debug mode or uncomment the following two lines and specify the output csv path
out_path = r'C:/Users/Peter/Downloads/ROP_Summary.csv'
summary.to_csv(out_path)

# Calculate median ROP value for entire dataset for comparison
median_rop = np.median(data.rop)

# Print percent change
# Note: I changed "increase" and "decrease" in the results and used absolute value for negative changes.
q1_print = 'By keeping the drilling parameters within the first quartile ROP, we increase the average speed by ' + str(round((summary.rop[0]/median_rop-1.00)*100.00,4)) + '%.'
q2_print = 'By keeping the drilling parameters within the second quartile ROP, we increase the average speed by ' + str(round((summary.rop[1]/median_rop-1.00)*100.00,2)) + '%.'
q3_print = 'By keeping the drilling parameters within the third quartile ROP, we decrease the average speed by ' + str(abs(round((summary.rop[2]/median_rop-1.00)*100.00,2))) + '%.'
q4_print = 'By keeping the drilling parameters within the fourth quartile ROP, we decrease the average speed by ' + str(abs(round((summary.rop[3]/median_rop-1.00)*100.00,2))) + '%.'
print(q1_print)
print(q2_print)
print(q3_print)
print(q4_print)

# TODO: Lump groups of the timestamps and divide by n to get a decimal value. Then run the quartiles again.

print('Done')