# initialData

# Cleans Data:
    # Gets rid of time difference data when it is the first data taken of the day (time is NaN or too large)
    # Gets rid of data that has a break of over 15 minutes
    # Gets rid of data when the drill is being pulled out
# Calculates and plots MSE vs ROP
# Produced Preliminary ML and does a feature importance analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
import math
import random
import statistics as st
import warnings
warnings.filterwarnings('ignore')


# Input and format Data ---------------------------------------------------
# Import Data
dataset = pd.read_csv(r'C:\Users\Peter\Downloads\EdgeProductivityAdvisorDemo.csv', encoding = "UTF-16-le")
dataset = dataset.iloc[::-1] # Reverse order, more intuitive in my opinion
dataset = dataset.reset_index(drop = True)

# Format time data
dataset.TimeStamp = dataset.TimeStamp.apply(lambda x: str(x)[:-3]) # Gets rid of "AM" or "PM" from time
dataset['DatesSplit'] = dataset['TimeStamp'].apply(lambda x: re.split('\s', x)) # Splits up day and time
dataset['Date'] = dataset.DatesSplit.apply(lambda x: x[0])
dataset['Time'] = dataset.DatesSplit.apply(lambda x: x[1])
dataset.TimeStamp = dataset.TimeStamp.apply(lambda x: datetime.strptime(x,'%m/%d/%Y %H:%M:%S')) # Makes it time variable


# Loop through each date in the dataset separately (that way we don't get huge deltaTimes in between days)
dates_list = list(set(dataset['Date']))
df_full = pd.DataFrame()
for date in dates_list:
    drilling_date = date
    dataset_subset = dataset.loc[dataset['Date'] == drilling_date]

    # Calculate the difference between each time (deltaTime)
    dataset_subset['TimeDiff'] = dataset_subset.TimeStamp.diff()
    dataset_subset['TimeDiff'] = dataset_subset['TimeDiff']-pd.to_timedelta(dataset_subset['TimeDiff'].dt.days, unit = 'd')
    dataset_subset['deltaTime'] = dataset_subset['TimeDiff'].apply(lambda x: x.seconds/60)
    dataset_subset = dataset_subset.drop(columns = ['DatesSplit', 'TimeDiff'])

    # Get rid of deltaTime data where the break is over 15 minutes (could be lunch, phone call, etc)
    def replaceBreaks(deltaTime):
        if deltaTime > 15:
            deltaTime = np.nan
        return deltaTime

    dataset_subset['deltaTime'] = dataset_subset['deltaTime'].apply(replaceBreaks)
    # Calculate the difference in drill string lengths to compare with detlaTime
    dataset_subset['deltaLength'] = dataset_subset['Drill String Length (ft)'].diff()
    dataset_subset['ROP (ft/hr)'] = (dataset_subset['deltaLength']/dataset_subset['deltaTime']) * 60
    dataset_subset['ROP (ft/min)'] = (dataset_subset['deltaLength'] / dataset_subset['deltaTime'])
    # Add each loop iteration together into a larger df_full
    df_full = df_full.append(dataset_subset)

# Got rid of 10/2? Probably outliers...
df_full = df_full.loc[df_full['Date'] != "10/2/2020"]

# Only took values where ROP is positive
df_full = df_full.loc[df_full['ROP (ft/hr)'] > 0]
df_full = df_full.reset_index(drop = True)

# MSE Calculation + Plot -------------------------------------------------
bit_diameter = 4.0
Area = math.pi*(bit_diameter/2.0)**2.0
df_full['MSE'] = (df_full['Thrust Force Max (lbf)']/Area)+((2*math.pi*df_full['Rotation Speed Max (rpm)']*df_full['Rotation Torque Max (ft-lb)'])/(Area*df_full['ROP (ft/hr)']))

# Save df_full to a csv to import to MATLAB (for Curve Fitting App)
# df_full.to_csv(r'C:\Users\Peter\Downloads\ROPvMSE.csv')

# Create profiling report (couldn't get it to work)
# from pandas_profiling import ProfileReport
# prof = ProfileReport(df_full)
# prof.to_file(output_file=r'C:\Users\Peter\Downloads\output.html')

# MSE vs ROP Graph with best-fits
# import numpy.polynomial.polynomial as poly
# x_new = np.linspace(min(df_full['MSE']), max(df_full['MSE']), 10000)

# coefs1, res1 = poly.polyfit(df_full['MSE'], df_full['ROP (ft/hr)'], 1, full = True)
# coefs2, res2 = poly.polyfit(df_full['MSE'], df_full['ROP (ft/hr)'], 2, full = True)
# coefs3, res3 = poly.polyfit(df_full['MSE'], df_full['ROP (ft/hr)'], 3, full = True)
#
# ffit1 = poly.polyval(x_new, coefs1)
# ffit2 = poly.polyval(x_new, coefs2)
# ffit3 = poly.polyval(x_new, coefs3)
#
# plt.plot(df_full['MSE'], df_full['ROP (ft/hr)'], 'o', label = 'MSE vs. ROP')
# plt.xlabel('Mechanical Specific Energy/MSE (psi)')
# plt.ylabel('ROP (ft/hr)')
# plt.title('MSE vs ROP')
# plt.plot(x_new, ffit1, 'g', label = '1st Degree Polyfit')
# plt.plot(x_new, ffit2, 'k', label = '2nd Degree Polyfit')
# plt.plot(x_new, ffit3, 'r', label ='3rd Degree Polyfit')
# plt.legend(loc = 'best')
# plt.savefig(r"C:\Users\Peter\Downloads\MSEvsROP")
# plt.show()


# Displays graphs for each variable vs. ROP
# for variable in df_full.columns:
#     plt.plot(df_full[str(variable)], df_full['ROP (ft/hr)'], 'o')
#     plt.xlabel(str(variable))
#     plt.ylabel('ROP (ft/hr)')
#     string_variable = variable.replace('/', '')
#     path = 'C:/Users/Peter/Downloads/' + string_variable + ' vs ROP.png'
#
#     plt.savefig(path)
#     plt.show()


# Physics based model validation ------------------------------------------
bit_diameter = 6
friction_coeff = 0.2
from PFModel import PFModel

# Caluclate ROP with PFModel and compare to actual ROP
df_full['PFModel'] = df_full.apply(lambda x: PFModel(x['Rotation Speed Max (rpm)'], x['Rotation Torque Max (ft-lb)'],
                                                     bit_diameter, friction_coeff), axis=1)
df_full['diff'] = df_full['ROP (ft/hr)']-df_full['PFModel']
df_full.reset_index()
# plt.plot(df_full.index, df_full.PFModel, 'r')
# plt.plot(df_full.index, df_full['ROP (ft/hr)'], 'k')
#plt.plot(df_full.index, df_full.MSE-2000)
#plt.show()

# mse = sum(df_full['diff'])/len(df_full)
from sklearn.metrics import mean_absolute_error
# mae = mean_absolute_error(df_full['ROP (ft/hr)'], df_full['PFModel'])
# print(mae)


# Machine Learning Model (Extra Trees) ------------------------------------
y = df_full['ROP (ft/hr)']
X = df_full.drop(columns = ['TimeStamp', 'Rod Count', 'ROP (ft/hr)', 'Date', 'Time', 'deltaTime', 'deltaLength',
                            'ROP (ft/min)', 'MSE', 'PFModel', 'diff'])

# Prints out MAE for each model
# from ModelAnalyzer import ModelAnalyzer
# ModelAnalyzer(X,y)

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X, y, train_size=0.7, shuffle=False)
etr = ExtraTreesRegressor(n_estimators=125)
etr.fit(train_X, train_Y)
etr_test_predictions = etr.predict(test_X)
etr_mae = mean_absolute_error(etr_test_predictions, test_Y)

importances = etr.feature_importances_
std = np.std([tree.feature_importances_ for tree in etr.estimators_],
             axis=0)

# Print the feature ranking
print("Feature ranking:")
for i,v in enumerate(importances):
    print('Feature: %0d, Score: %.5f' % (i,v))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances,
        color="r", yerr=std, align="center")
plt.xticks(range(X.shape[1]))
plt.xlim([-1, X.shape[1]])
plt.show()
print('Done')