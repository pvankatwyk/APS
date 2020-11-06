# Regressor ML Model
# This model:
#   1. Creates a synthetic dataset based on upper and lower boundaries of thrust, rpm, torque, bit diameter, friction
#   2. Runs the ZT Model (Zacny, Teale) and calculates ROP
#   3. Preforms basic model selection, predicting the friction coef. based on thrust, rpm, torque, bit diameter, and ROP
#   4. Runs an Extra Trees Regressor model
#   5. Calculates mean absolute error as a form of validation

# Regressor ML Model
# Steps:
#   1. Makes a synthetic dataset based on input parameters (see "makeDataset.py")
#   2. Determines which ML model has the lowest mean absolute error when predicting friction coefficient
#   3. Runs the optimal model - in this case an Extra Trees model
#   4. Classifies the predicted coefficients into soil type and compares to original coefficient classification
#   5. Calculates proportion correct (accuracy score) as a form of validation

# Results:
#   - More accurate and faster than MLClassifier
#   - ZTModel generally more accuracy most likely because it is a simpler model form
#   - ZTModel - At 100,000 samples, on average the model can predict the friction coefficient within 0.0092. After
#           predicting then classifying this way, the proportion correct is 0.89617
#   - PFModel - At 100,000 samples, on average the model can predict the friction coefficient within 0.0143. After
#           predicting then classifying this way, the proportion correct is 0.84403

import time
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from makeDataset import makeDataset, soilClassifier
import warnings

warnings.filterwarnings('ignore')
np.random.seed(10)

# ------------------- CREATE DATASET -------------------
# Set upper and lower bounds for synthetic dataset parameters
dataset_length = 100000
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

dataset = makeDataset(dataset_length, RPM_lower, RPM_upper, torque_lower, torque_upper, thrust_lower, thrust_upper,
                      bit_lower, bit_upper, friction_lower, friction_upper)

# ------------------- MAKE ML MODEL -------------------

# Establish features/labels and make train and test datasets
X = dataset[['thrust', 'torque', 'RPM', 'bit_diameter', 'PF_ROP']]
y = dataset.friction_coeff

# Split dataset into train and test dataset (train_size is the proportion of train to test lengths)
train_X, test_X, train_Y, test_Y = train_test_split(X, y, train_size=0.7, shuffle=False, random_state=1)

# Run several models and determine prediction accuracy using accuracy score.

# Model Selection
# Decision Tree
# start_dt = time.time()
# dt = DecisionTreeRegressor(random_state=1)
# dt.fit(train_X, train_Y)
# dt_test_predictions = dt.predict(test_X)
# dt_mae = mean_absolute_error(dt_test_predictions, test_Y)
# finish_dt = str(round(time.time() - start_dt, 5))
# out = "Decision Tree MAE: " + str(dt_mae) + ', Time: ' + str(finish_dt) + ' seconds.'
# print(out)
# Result ZTModel - Decision Tree MAE: 0.0387977250120633, Time: 0.11744 seconds. (dataset_length = 25,000)
# Result PFModel - Decision Tree MAE: 0.05791211426534548, Time: 0.09549 seconds.

# Random Forest
# start_rf = time.time()
# rf = RandomForestRegressor(random_state=1, n_estimators=100)
# rf.fit(train_X, train_Y)
# rf_test_predictions = rf.predict(test_X)
# rf_mae = mean_absolute_error(rf_test_predictions, test_Y)
# finish_rf = str(round(time.time() - start_rf, 5))
# out = "Random Forest MAE: " + str(rf_mae) + ', Time: ' + str(finish_rf) + ' seconds.'
# print(out)
# Result ZTModel - Random Forest MAE: 0.02492721433495952, Time: 11.75553 seconds. (dataset_length = 25,000)
# Result PFModel - Random Forest MAE: 0.037157794165862436, Time: 6.01006 seconds.

# Support Vector Regressor
# start_svr = time.time()
# svr = SVR(gamma='scale', C=1.0)
# svr.fit(train_X, train_Y)
# svr_test_predictions = svr.predict(test_X)
# svr_mae = mean_absolute_error(svr_test_predictions, test_Y)
# finish_svr = str(round(time.time() - start_svr, 5))
# out = "Support Vector MAE: " + str(svr_mae) + ', Time: ' + str(finish_svr) + ' seconds.'
# print(out)
# Result ZTModel - Support Vector MAE: 0.10351615206185287, Time: 77.97592 seconds. (dataset_length = 25,000)
# Result PFModel - Support Vector MAE: 0.09726490727097015, Time: 35.12777 seconds.


# EXTRA TREES MODEL
start_etr = time.time()
etr = ExtraTreesRegressor(n_estimators=125)
etr.fit(train_X, train_Y)
etr_test_predictions = etr.predict(test_X)
etr_mae = mean_absolute_error(etr_test_predictions, test_Y)
finish_etr = str(round(time.time() - start_etr, 5))
out = "Extra Trees MAE: " + str(etr_mae) + ', Time: ' + str(finish_etr) + ' seconds.'
print(out)
# Result ZTModel - Extra Trees MAE: 0.018575455978610687, Time: 5.69328 seconds. (dataset_length = 25,000)
# Result PFModel - Extra Trees MAE: 0.028567458269551586, Time: 6.25179 seconds.

# Make Dataframe with results
results_DF = pd.DataFrame()
results_DF['Test_Y'] = test_Y
results_DF['Classification'] = dataset['classification']
# results_DF['DecisionTree Diff'] = dt_test_predictions
# results_DF['RandomForest Diff'] = rf_test_predictions
# results_DF['SVR Diff'] = svr_test_predictions
results_DF['ExtraTrees'] = etr_test_predictions

# ------------------- PARAMETER OPTIMIZATION -------------------
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
# print(gsc.best_estimator_)

# Classify the new
results_DF['classification_pred'] = results_DF['ExtraTrees'].apply(lambda x: soilClassifier(x))
# Uncomment next line and change file path to export results
# results_DF.to_csv(r'C:\Users\Peter\Downloads\accuracy.csv')

# Calculate accuracy score (proportion correct) from categorical responses
etr_accuracy = accuracy_score(results_DF['Classification'], results_DF['classification_pred'])
out = 'After predicting the coefficient then classifying, the accuracy score is: ' + str(round(etr_accuracy, 5))
print(out)

# Example of predicting soil type based on parameters:
# test_example = {'thrust': [1200], 'torque': [1000], 'RPM': [150], 'bit_diameter': [4], 'ZT_ROP': [36]}
# test_df = pd.DataFrame(data = test_example)
# soil_type_test = soilClassifier(etr.predict(test_df))
# out = 'Test Example - Thrust: ' + str(test_example['thrust']) + ', Torque: ' + str(test_example['torque']) + \
#       ', RPM: ' + str(test_example['RPM']) + ', Bit Diameter: ' + str(test_example['bit_diameter']) + ', ROP: ' + \
#       str(test_example['ZT_ROP']) + ' -- Soil Type: ' + str(soil_type_test) + '.'
# print(out)
