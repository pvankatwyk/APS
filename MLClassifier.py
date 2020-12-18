# Classifier ML Model
# Steps:
#   1. Makes a synthetic dataset based on input parameters (see "makeDataset.py")
#   2. Determines which ML model has the highest proportion of soil type predictions correct
#   4. Runs the optimal model - in this case a Random Forest model
#   5. Calculates proportion correct (accuracy score) as a form of validation
# Note:This particular method generated friction coefficients and at the beginning, classified them into their
# respective soil types. The categorical variable storing the soil type classification (see makeDataset.py) was used
# as the label which required a classifier rather than a regressor.

# Results:
#   - Not as accurate as the MLRegressor -- it did much better predicting quantities than categories
#   - ZTModel - At 100,000 samples in the synthetic dataset, the accuracy was 0.7335 and took roughly 65 seconds
#   - PFModel - At 100,000 samples in the synthetic dataset, the accuracy was 0.6379 and took roughly 35 seconds
#   - We would be better off predicting the friction coefficient with the model followed by the soil classification

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import warnings
from makeDataset import makeDataset
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
np.random.seed(10)


# ------------------- CREATE DATASET -------------------
# Set upper and lower bounds for synthetic dataset parameters
dataset_length = 25000
RPM_lower = 50
RPM_upper = 500
torque_lower = 50
torque_upper = 2500
thrust_lower = 50
thrust_upper = 2500
bit_lower = 3
bit_upper = 7
friction_lower = 0.4
friction_upper = 0.9

dataset = makeDataset(dataset_length, RPM_lower, RPM_upper, torque_lower, torque_upper, thrust_lower, thrust_upper,
                      bit_lower,bit_upper, friction_lower, friction_upper)

# ------------------- MAKE ML MODEL -------------------

# Establish features/labels and make train and test datasets
X = dataset[['thrust', 'torque', 'RPM', 'bit_diameter', 'PF_ROP']] # Features
y = dataset.classification # Label

# Split dataset into train and test dataset (train_size is the proportion of train to test lengths)
train_X, test_X, train_Y, test_Y = train_test_split(X, y, train_size=0.7,shuffle=False, random_state=1)


from Subfunctions import ModelAnalyzer
ModelAnalyzer(X,y, False)
# Run several models and determine prediction accuracy using accuracy score.

# Logistic Regression
# from sklearn.linear_model import LogisticRegression
# start = time.time()
# lr = LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=2000)
# lr.fit(train_X, train_Y)
# lr_predictions = lr.predict(test_X)
# finish_lr = str(round(time.time()-start,5))
# lr_accuracy = accuracy_score(test_Y, lr_predictions)
# out = "Logistic Regression Accuracy: " + str(lr_accuracy) + ', Time: ' + str(finish_lr) + ' seconds.'
# print(out)
# Result ZTModel - Logistic Regression Accuracy: 0.25253333333333333, Time: 12.84606 seconds. (dataset_length = 25,000)
# Result PFModel - Logistic Regression Accuracy: 0.3410666666666667, Time: 16.70157 seconds.

# Naïve Bayes
# from sklearn.naive_bayes import GaussianNB
# start = time.time()
# nb = GaussianNB()
# nb.fit(train_X, train_Y)
# nb_predictions = nb.predict(test_X)
# finish_nb = str(round(time.time()-start,5))
# nb_accuracy = accuracy_score(test_Y, nb_predictions)
# out = "Naive Bayes Accuracy: " + str(nb_accuracy) + ', Time: ' + str(finish_nb) + ' seconds.'
# print(out)
# Result ZTModel - Naive Bayes Accuracy: 0.21813333333333335, Time: 0.02997 seconds. (dataset_length = 25,000)
# Result PFModel - Naive Bayes Accuracy: 0.26613333333333333, Time: 0.0369 seconds.

# Stochastic Gradient Descent
# from sklearn.linear_model import SGDClassifier
# start = time.time()
# sgd = SGDClassifier(loss='modified_huber', shuffle=True,random_state=101,tol=1e-3,max_iter=1000)
# sgd.fit(train_X, train_Y)
# sgd_predictions = sgd.predict(test_X)
# finish_sgd = str(round(time.time()-start,5))
# sgd_accuracy = accuracy_score(test_Y, sgd_predictions)
# out = "SGD Accuracy: " + str(sgd_accuracy) + ', Time: ' + str(finish_sgd) + ' seconds.'
# print(out)
# Result ZTModel - SGD Accuracy: 0.0984, Time: 3.15821 seconds. (dataset_length = 25,000)
# Result PFModel - SGD Accuracy: 0.13813333333333333, Time: 2.56967 seconds.

# K-Nearest Neighbors
# from sklearn.neighbors import KNeighborsClassifier
# start = time.time()
# knn = KNeighborsClassifier(n_neighbors=10)
# knn.fit(train_X, train_Y)
# knn_predictions = knn.predict(test_X)
# finish_knn = str(round(time.time()-start,5))
# knn_accuracy = accuracy_score(test_Y, knn_predictions)
# out = "KNN Accuracy: " + str(knn_accuracy) + ', Time: ' + str(finish_knn) + ' seconds.'
# print(out)
# Result ZTModel - KNN Accuracy: 0.198, Time: 0.12955 seconds. (dataset_length = 25,000)
# Result PFModel - KNN Accuracy: 0.2608, Time: 0.16562 seconds.

# Decision Tree
# from sklearn.tree import DecisionTreeClassifier
# start = time.time()
# dt = DecisionTreeClassifier(max_depth=10,random_state=101,max_features=None,min_samples_leaf=5)
# dt.fit(train_X, train_Y)
# dt_predictions = dt.predict(test_X)
# finish_dt = str(round(time.time()-start,5))
# dt_accuracy = accuracy_score(test_Y, dt_predictions)
# out = "Decision Tree Accuracy: " + str(dt_accuracy) + ', Time: ' + str(finish_dt) + ' seconds.'
# print(out)
# Result ZTModel - Decision Tree Accuracy: 0.3762666666666667, Time: 0.11059 seconds. (dataset_length = 25,000)
# Result PFModel - Decision Tree Accuracy: 0.3808, Time: 0.13881 seconds.

# Random Forest
start = time.time()
rfm = RandomForestClassifier(n_estimators=125,oob_score=True,n_jobs=1,random_state=101,max_features=None,min_samples_leaf=3)
rfm.fit(train_X, train_Y)
rfm_predictions = rfm.predict(test_X)
finish_rfm = str(round(time.time()-start,5))
rfm_accuracy = accuracy_score(test_Y, rfm_predictions)
out = "Random Forest Accuracy: " + str(rfm_accuracy) + ', Time: ' + str(finish_rfm) + ' seconds.'
print(out)
# Result ZTModel - Random Forest Accuracy: 0.6146666666666667, Time: 13.61088 seconds. (dataset_length = 25,000)
# Result PFModel - Random Forest Accuracy: 0.514, Time: 7.98569 seconds.

# Support Vector Classifier
# from sklearn.svm import SVC
# start = time.time()
# svm = SVC(gamma='scale', C=1.0, random_state=101)
# svm.fit(train_X, train_Y)
# svm_predictions = svm.predict(test_X)
# finish_svm = str(round(time.time()-start,5))
# svm_accuracy = accuracy_score(test_Y, svm_predictions)
# out = "SVC Accuracy: " + str(svm_accuracy) + ', Time: ' + str(finish_svm) + ' seconds.'
# print(out)
# Result ZTModel - SVC Accuracy: 0.2132, Time: 60.1203 seconds. (dataset_length = 25,000)
# Result PFModel - SVC Accuracy: 0.26066666666666666, Time: 51.14697 seconds.

# Extra Trees
# from sklearn.ensemble import ExtraTreesClassifier
# start = time.time()
# etc = ExtraTreesClassifier(n_estimators=125)
# etc.fit(train_X, train_Y)
# etc_predictions = etc.predict(test_X)
# finish_etc = str(round(time.time()-start, 5))
# etc_accuracy = accuracy_score(test_Y, etc_predictions)
# out = "Extra Trees Accuracy: " + str(etc_accuracy) + ', Time: ' + str(finish_etc) + ' seconds.'
# print(out)
# Result ZTModel- Extra Trees Accuracy: 0.5508, Time: 4.16618 seconds. (dataset_length = 25,000)
# Result PFModel - Extra Trees Accuracy: 0.484, Time: 3.2097 seconds.

# Make Dataframe with results
results_DF = pd.DataFrame()
results_DF['Test_Y'] = test_Y
# results_DF['Naive Bayes'] = nb_predictions
# results_DF['SGD'] = sgd_predictions
# results_DF['K-NN'] = knn_predictions
# results_DF['DecisionTree'] = dt_predictions
results_DF['RandomForest'] = rfm_predictions
# results_DF['SVC'] = svm_predictions
# results_DF['ExtraTrees'] = etc_predictions

# If you run all models, uncomment the next line to output a csv with the results
# results_DF.to_csv(r'C:\Users\Peter\Downloads\results.csv')


# Note: Use GridSearchCV or Hyperopt to get best model parameters