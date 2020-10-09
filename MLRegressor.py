# Create a Machine Learning model based on generated data from existing physics-based models

import numpy as np
import pandas as pd
from ZTModel import ZTModel
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
import random
import time
import warnings
warnings.filterwarnings('ignore')
np.random.seed(10)


# DATASET PARAMETERS___________________________________________________________
# Set upper and lower bounds for parameters
dataset_length = 20000
RPM_lower = 50
RPM_upper = 500
torque_lower = 50
torque_upper = 2500
thrust_lower = 50
thrust_upper = 2500
bit_lower = 3
bit_upper = 7
friction_lower = 0.5
friction_upper = 0.9

# MAKE SYNTHETIC DATASET_________________________________________________________
RPM = np.random.randint(RPM_lower, RPM_upper+1, dataset_length)
torque = np.random.randint(torque_lower, torque_upper+1, dataset_length)
thrust = np.random.randint(thrust_lower, thrust_upper+1, dataset_length)
bit_diameter = np.random.randint(bit_lower, bit_upper + 1, dataset_length)
friction_coeff = []
for i in range(dataset_length):
    friction_coeff.append(friction_lower + (friction_upper-friction_lower)*random.random())

dataset = pd.DataFrame()
dataset['thrust'] = thrust
dataset['torque'] = torque
dataset['RPM'] = RPM
dataset['bit_diameter'] = bit_diameter
dataset['friction_coeff'] = friction_coeff

dataset['ZTModel_ROP'] = dataset.apply(lambda x: ZTModel(x.torque, x.thrust, x.bit_diameter, x.friction_coeff), axis=1)
#dataset['PFModel_ROP'] = dataset.apply(lambda x: PFModel(x.RPM, x.thrust, x.bit_diameter, x.friction_coeff), axis=1)

# MAKE MACHINE LEARNING MODEL____________________________________________________
X = dataset[['thrust', 'torque', 'RPM', 'bit_diameter', 'ZTModel_ROP']]
y = dataset.friction_coeff

train_X, test_X, train_Y, test_Y = train_test_split(X, y, train_size=0.7,shuffle=False, random_state=1)

# Model Selection
# start_dt = time.time()
# dt = DecisionTreeRegressor(random_state=1)
# dt.fit(train_X, train_Y)
# dt_test_predictions = dt.predict(test_X)
# dt_mae = mean_absolute_error(dt_test_predictions, test_Y)
# finish_dt = str(round(time.time() - start_dt, 5))
# out = "Decision Tree MAE: " + str(dt_mae) + ', Time: ' + str(finish_dt) + ' seconds.'
# print(out)
#
# start_rf = time.time()
# rf = RandomForestRegressor(random_state=1, n_estimators=100)
# rf.fit(train_X, train_Y)
# rf_test_predictions = rf.predict(test_X)
# rf_mae = mean_absolute_error(rf_test_predictions, test_Y)
# finish_rf = str(round(time.time() - start_rf, 5))
# out = "Random Forest MAE: " + str(rf_mae) + ', Time: ' + str(finish_rf) + ' seconds.'
# print(out)
#
# start_svr = time.time()
# svr = SVR(gamma = 'scale', C = 1.0)
# svr.fit(train_X, train_Y)
# svr_test_predictions = svr.predict(test_X)
# svr_mae = mean_absolute_error(svr_test_predictions, test_Y)
# finish_svr = str(round(time.time() - start_svr, 5))
# out = "Support Vector MAE: " + str(svr_mae) + ', Time: ' + str(finish_svr) + ' seconds.'
# print(out)

start_etr = time.time()
etr = ExtraTreesRegressor(n_estimators = 125)
etr.fit(train_X, train_Y)
etr_test_predictions = etr.predict(test_X)
etr_mae = mean_absolute_error(etr_test_predictions, test_Y)
finish_etr = str(round(time.time() - start_etr, 5))
out = "Extra Trees MAE: " + str(etr_mae) + ', Time: ' + str(finish_etr) + ' seconds.'
print(out)


results_DF = pd.DataFrame()
results_DF['Test_Y'] = test_Y
# results_DF['DecisionTree Diff'] = dt_test_predictions
# results_DF['RandomForest Diff'] = rf_test_predictions
# results_DF['SVR Diff'] = svr_test_predictions
results_DF['ExtraTrees_Diff Diff'] = etr_test_predictions


# for estimator in [50,60,70,80,90,100,125,150,175,200]:
#     start = time.time()
#     etr = ExtraTreesRegressor(random_state = 1, n_estimators=estimator)
#     etr.fit(train_X, train_Y)
#     etr_test_predictions = etr.predict(test_X)
#     etr_mae = mean_absolute_error(etr_test_predictions, test_Y)
#     finish = str(round(time.time() - start, 5))
#     out = 'N_Estimators: ' + str(estimator) + ', MAE: ' + str(etr_mae) + ', Time: ' + finish + '.'
#     print(out)

# gsc = GridSearchCV(
#     estimator=etr,
#     param_grid={
#         'n_estimators': range(50,126,10),
#         'max_features': range(1,11, 1),
#         'min_samples_leaf': range(2,5,1),
#         'min_samples_split': range(2,5,1),
#     },
#     scoring='r2',
#     cv=5,
#     n_jobs=-1,
#     verbose=1
# )
# gsc.fit(train_X, train_Y)


# Make the friction coefficient not a nice number, then assign it to one of the soil types, then put in the original
# dataframe what kind of soil that is, then use classifiers instead of regressors
# OR -- classify them later, so use regressors but use the prediction to then put them into bins



print('Done')
