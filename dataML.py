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
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import warnings
from Subfunctions import ModelAnalyzer
from Subfunctions import PrepareData
from Subfunctions import ParamOptimize
warnings.filterwarnings('ignore')

# Prepares data from data folder
data = PrepareData(folder = r'C:/Users/Peter/Documents/APS/data/', profiling = False)

# Add soil data and do an inner join with current data on location
soilData = pd.read_csv(r'C:/Users/Peter/Downloads/soils.csv')
data = data.merge(soilData, left_on=data['Location'], right_on=soilData['Location'])
data.drop(columns = ['key_0', 'Location_x', 'Location_y', 'Soil Type'], inplace = True)

# Pick features and targets
X = data.drop(columns = ['ROP (ft/min)', 'Drill', 'Time', 'Rod Count', 'ROP (ft/hr)'])
# X = data.drop(columns = ['ROP (ft/min)', 'Drill', 'Time', 'Rod Count', 'ROP (ft/hr)', 'Thrust Speed Avg (ft/min)', 'Pull Force Maximum (lbf)', 'Pull Speed Average (ft/min)'])
y = data['ROP (ft/min)']
ModelAnalyzer(X,y, regressor = True)

# Split into train and test datasets
train_X, test_X, train_Y, test_Y = train_test_split(X, y, train_size=0.5, shuffle=False, random_state=1)

# EXTRA TREES MODEL
start_etr = time.time()
etr = ExtraTreesRegressor(n_estimators=125)
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

# importances = etr.feature_importances_
# std = np.std([tree.feature_importances_ for tree in etr.estimators_],
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


print('Done')