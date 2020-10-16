# Model Validation
# This script:
#   1. Makes a synthetic dataset (like other ML scripts)
#   2. Runs the Classifier ML algorithm at different dataset lengths (see "length_list")
#   3. Runs the Regressor ML algorithm at different dataset lengths (see "length_list")
#   4. Compares and plots the accuracy and analysis time of each model

# Results:
#   - It is faster and more accurate to predict using a regressor.
#   - Regressor starts to level out at around 50,000-60,000 length. Accuracy score at 100,000 is ~0.90
#   - Classifier levels out around the same, time. Accuracy score at 100,000 is ~0.70
#   - Time for both is relatively linear, with the regressor being much faster
#           Regressor Time @ 100,000 -- ~20 seconds
#           Classifier Time @ 100,000 -- ~40 seconds

import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from makeDataset import makeDataset, soilClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# dataset_length = 50000
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


# CLASSIFIER
# Establish a list for all the different dataset_lengths to test
length_list = [100,250,500,1000,2500, 5000, 7500, 10000, 15000, 20000, 30000, 40000, 50000, 75000, 100000]
#length_list = [100,200,300]
time_list_classifier = []
acc_list_classifier = []
# Run the Random Forest Model for every dataset_length in length_list
for dataset_length in length_list:
    dataset = makeDataset(dataset_length, RPM_lower, RPM_upper, torque_lower, torque_upper, thrust_lower, thrust_upper,
                          bit_lower, bit_upper, friction_lower, friction_upper)
    X = dataset[['thrust', 'torque', 'RPM', 'bit_diameter', 'ZT_ROP']]
    y = dataset.classification
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, train_size=0.7, shuffle=False, random_state=1)
    start = time.time()
    rfm = RandomForestClassifier(n_estimators=125,oob_score=True,n_jobs=1,random_state=101,max_features=None,min_samples_leaf=3)
    rfm.fit(train_X, train_Y)
    rfm_predictions = rfm.predict(test_X)
    finish_rfm = round(time.time()-start,5)
    acc_list_classifier.append(accuracy_score(test_Y, rfm_predictions)) # Appends accuracy to a list of all lengths
    time_list_classifier.append(finish_rfm) # Appends analysis time to a list of all lengths


# REGRESSOR
# Establish a list for all the different dataset_lengths to test
time_list_regressor = []
acc_list_regressor = []
# Run the Extra Trees Model for every dataset_length in length_list
for dataset_length in length_list:
    dataset = makeDataset(dataset_length, RPM_lower, RPM_upper, torque_lower, torque_upper, thrust_lower, thrust_upper,
                          bit_lower, bit_upper, friction_lower, friction_upper)
    X = dataset[['thrust', 'torque', 'RPM', 'bit_diameter', 'ZT_ROP']]
    y = dataset.friction_coeff
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, train_size=0.7, shuffle=False, random_state=1)
    start_etr = time.time()
    etr = ExtraTreesRegressor(n_estimators=125)
    etr.fit(train_X, train_Y)
    etr_test_predictions = etr.predict(test_X)
    finish_etr = (round(time.time() - start_etr, 5))
    results_DF = pd.DataFrame()
    results_DF['Test_Y'] = test_Y
    results_DF['Classification'] = dataset['classification']
    results_DF['ExtraTrees'] = etr_test_predictions
    classification_pred = results_DF['ExtraTrees'].apply(lambda x: soilClassifier(x))
    etr_accuracy = accuracy_score(results_DF['Classification'], classification_pred)
    acc_list_regressor.append(etr_accuracy)
    time_list_regressor.append(finish_etr)

# Create a plot that shows:
#   - Subplot 1: Accuracy score vs Dataset_length
#   - Subplot 2: Analysis time vs Dataset_length
fig,axs = plt.subplots(2, sharex = True)
axs[0].plot(length_list, acc_list_classifier, 'g', label = 'Classifier (Random Forest)')
axs[0].plot(length_list, acc_list_regressor, 'r', label = 'Regressor (Extra Trees)')
axs[0].set_title('Model Accuracy Score vs Dataset Size')
axs[0].set_ylabel('Proportion Correct')
axs[0].legend(fancybox = True)

axs[1].plot(length_list, time_list_classifier, 'g', label = 'Classifier')
axs[1].plot(length_list, time_list_regressor, 'r', label = 'Regressor')
axs[1].set_title('Model Analysis Time vs Dataset Size')
axs[1].set_xlabel('Size of Synthetic Dataset')
axs[1].set_ylabel('Time to fit (seconds)')

plt.show()