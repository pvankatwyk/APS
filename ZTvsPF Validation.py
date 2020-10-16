# ZT vs PF
# This script:
#   1. Makes a synthetic dataset (like other ML scripts)
#   2. Runs the Classifier ML algorithm at different dataset lengths for both the PFModel and ZTModel
#   3. Runs the Regressor ML algorithm at different dataset lengths for both the PFModel and ZTModel
#   4. Compares and plots the accuracy of the PFModel/ZTModel using a Regressor and Classifier

# Results:
#   - The PFModel has lesser accuracy due to the increased complexity of the model itself
#   - At 100,000 samples, the accuracy between the two are close

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
# length_list = [100,200,300, 1000]
time_list_PF = []
acc_list_PF = []
acc_list_class_PF = []

# Run the Random Forest Model for every dataset_length in length_list
for dataset_length in length_list:
    dataset = makeDataset(dataset_length, RPM_lower, RPM_upper, torque_lower, torque_upper, thrust_lower, thrust_upper,
                          bit_lower, bit_upper, friction_lower, friction_upper)
    X = dataset[['thrust', 'torque', 'RPM', 'bit_diameter', 'PF_ROP']]
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
    acc_list_PF.append(etr_accuracy) # Appends accuracy to a list of all lengths
    time_list_PF.append(finish_etr) # Appends analysis time to a list of all lengths

    y = dataset.classification
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, train_size=0.7, shuffle=False, random_state=1)
    start = time.time()
    rfm = RandomForestClassifier(n_estimators=125,oob_score=True,n_jobs=1,random_state=101,max_features=None,min_samples_leaf=3)
    rfm.fit(train_X, train_Y)
    rfm_predictions = rfm.predict(test_X)
    finish_rfm = round(time.time()-start,5)
    acc_list_class_PF.append(accuracy_score(test_Y, rfm_predictions)) # Appends accuracy to a list of all lengths
    #time_list_classifier.append(finish_rfm) # Appends analysis time to a list of all lengths




# REGRESSOR
# Establish a list for all the different dataset_lengths to test
time_list_ZT = []
acc_list_ZT = []
acc_list_class_ZT = []
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
    acc_list_ZT.append(etr_accuracy)
    time_list_ZT.append(finish_etr)

    y = dataset.classification
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, train_size=0.7, shuffle=False, random_state=1)
    start = time.time()
    rfm = RandomForestClassifier(n_estimators=125,oob_score=True,n_jobs=1,random_state=101,max_features=None,min_samples_leaf=3)
    rfm.fit(train_X, train_Y)
    rfm_predictions = rfm.predict(test_X)
    finish_rfm = round(time.time()-start,5)
    acc_list_class_ZT.append(accuracy_score(test_Y, rfm_predictions)) # Appends accuracy to a list of all lengths
    #time_list_classifier.append(finish_rfm) # Appends analysis time to a list of all lengths

# Create a plot that shows:
#   - Subplot 1: Accuracy score vs Dataset_length
#   - Subplot 2: Analysis time vs Dataset_length
fig,axs = plt.subplots(2, sharex = True)
axs[0].plot(length_list, acc_list_PF, 'b', label = 'Pessier & Fear Model')
axs[0].plot(length_list, acc_list_ZT, color = 'orange', label = 'Zacney & Teale Model')
axs[0].set_title('REGRESSOR - Model Accuracy Score vs Dataset Size')
axs[0].set_ylabel('Proportion Correct')
axs[0].legend(loc='best', fancybox = True)

axs[1].plot(length_list, acc_list_class_PF, 'b', label = 'Pessier & Fear Model')
axs[1].plot(length_list, acc_list_class_ZT, color = 'orange', label = 'Zacney & Teale Model')
axs[1].set_title('CLASSIFIER - Model Accuracy Score vs Dataset Size')
axs[1].set_xlabel('Size of Synthetic Dataset')
axs[1].set_ylabel('Proportion Correct')

plt.show()