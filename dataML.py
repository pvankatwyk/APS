# dataML
# Takes data from data folder, prepares it, and fits it to the best ML model.

import time
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
from sklearn.tree import DecisionTreeRegressor
from Subfunctions import ModelAnalyzer
from Subfunctions import PrepareData
from Subfunctions import ParamOptimize
warnings.filterwarnings('ignore')

# Prepares data from data folder
data = PrepareData(folder = r'C:/Users/Peter/Documents/Work/APS/data/', profiling = False)

# Add soil data and do an inner join with current data on location
# soilData = pd.read_csv(r'C:/Users/Peter/Downloads/soils.csv')
# data = data.merge(soilData, left_on=data['Location'], right_on=soilData['Location'])
# data.drop(columns = ['key_0', 'Location_x', 'Location_y', 'Soil Type'], inplace = True)

# Pick features and targets
X = data.drop(columns = ['ROP (ft/min)', 'Drill', 'Time', 'Rod Count', 'ROP (ft/hr)'])
y = data['ROP (ft/min)']
# ModelAnalyzer(X,y, regressor = True)

# Split into train and test datasets
train_X, test_X, train_Y, test_Y = train_test_split(X, y, train_size=0.5, shuffle=False, random_state=1)

# EXTRA TREES MODEL
start_etr = time.time()
etr = ExtraTreesRegressor(n_estimators=125, random_state=1)
etr.fit(train_X, train_Y)
etr_test_predictions = etr.predict(test_X)
etr_mae = mean_absolute_error(etr_test_predictions, test_Y)
finish_etr = str(round(time.time() - start_etr, 5))


# Plot MSE
# data_subset = data[data['MSE']<10000] # Get rid of outliers
# data_subset.to_csv(r'C:\Users\Peter\Downloads\data_ML.csv', index = False)

# plt.plot(data_subset['MSE'], data_subset['ROP (ft/min)'], 'o', label = 'MSE vs. ROP')
# plt.xlabel('Mechanical Specific Energy/MSE (psi)')
# plt.ylabel('ROP (ft/hr)')
# plt.title('MSE vs ROP')
# plt.show()


# Feature Importances
# model = etr
# importances = model.feature_importances_
# std = np.std([tree.feature_importances_ for tree in model.estimators_],
#              axis=0)
# Print the feature ranking
# print("Feature ranking:")
# for i,v in enumerate(importances):
#     print('Feature: %0d, Score: %.5f' % (i,v))
#
# # Plot the impurity-based feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances,
#         color="r", yerr=std, align="center")
# plt.xticks(range(X.shape[1]))
# plt.xlim([-1, X.shape[1]])
# plt.show()



# Median Dataset Plots
maximum_params = [] # List to store values for maximizing ROP
for column in X.columns:
    delta_column = range(int(min(X[column])), int(max(X[column]))) # Create array for the variable being changed
    df = pd.DataFrame(delta_column)
    # Create columns for all variables with the median values
    df['Rotation Speed Max (rpm)'] = np.median(X['Rotation Speed Max (rpm)'])
    df['Rotation Torque Max (ft-lb)'] = np.median(X['Rotation Torque Max (ft-lb)'])
    df['Thrust Force Max (lbf)'] = np.median(X['Thrust Force Max (lbf)'])
    df['Mud Flow Rate Avg (gpm)'] = np.median(X['Mud Flow Rate Avg (gpm)'])
    df['Mud Pressure Max (psi)'] = np.median(X['Mud Pressure Max (psi)'])
    df['Thrust Speed Avg (ft/min)'] = np.median(X['Thrust Speed Avg (ft/min)'])
    df['Pull Force Maximum (lbf)'] = np.median(X['Pull Force Maximum (lbf)'])
    df['Pull Speed Average (ft/min)'] = np.median(X['Pull Speed Average (ft/min)'])
    df['Drill String Length (ft)'] = np.median(X['Drill String Length (ft)'])
    # Delete the median column for the column being changed
    del df[column]
    # Rename the delta_column array to the column of interest
    df = df.rename(columns={0: column})
    # Reorder the columns to the match model requirements
    df = df[
        ['Rotation Speed Max (rpm)', 'Rotation Torque Max (ft-lb)', 'Thrust Force Max (lbf)', 'Mud Flow Rate Avg (gpm)',
         'Mud Pressure Max (psi)', 'Thrust Speed Avg (ft/min)', 'Pull Force Maximum (lbf)', 'Pull Speed Average (ft/min)',
         'Drill String Length (ft)']]

    # Predict with ETR model and plot
    prediction = etr.predict(df)
    plt.plot(df[column], prediction)
    plt.xlabel(column)
    plt.ylabel('ROP (ft/min)')
    title = column + ' vs ROP'
    plt.title(title)
    plt.show()
    # Append value that produces the maximum ROP prediction value
    maximum_params.append(df[column][prediction.argmax()])

# Use all of the maximum values to predict the maximum ROP
maximum_params = np.array(maximum_params)
maximum_rop = etr.predict(maximum_params.reshape(1,-1))
# maximum_rop = 10 ft/min (which is the maximum ROP value in the dataset)

print('Done')