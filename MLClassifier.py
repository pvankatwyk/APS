# Regressor ML Model
# This model:
#   1. Creates a synthetic dataset based on upper and lower boundaries of thrust, rpm, torque, bit diameter, friction
#   2. Runs the ZT Model (Zacny, Teale) and calculates ROP
#   3. Preforms basic model selection, predicting the friction coef. based on thrust, rpm, torque, bit diameter, and ROP
#   4. Runs an Extra Trees Regressor model
#   5. Calculates mean absolute error as a form of validation

import numpy as np
import pandas as pd
from ZTModel import ZTModel
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import cross_val_score
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
friction_lower = 0.4
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

def soilClassifier(x):
    if x < 0.47:
        out = 'Organic Silt or Clay'
    elif x > 0.47 and x <= 0.59:
        out = 'Clay'
    elif x > 0.59 and x <= 0.63:
        out = 'Clayey Silt or Sand'
    elif x > 0.63 and x <= 0.65:
        out = 'Silt'
    elif x > 0.65 and x <= 0.69:
        out = 'Silty Sand'
    elif x > 0.69 and x <= 0.75:
        out = 'Silty Gravel'
    elif x > 0.75 and x <= 0.80:
        out = 'Fine to Course Sand'
    elif x > 0.8:
        out = 'Fine to Course Gravel'
    return out

dataset['classification'] = dataset['friction_coeff'].apply(lambda x: soilClassifier(x))


dataset['ZTModel_ROP'] = dataset.apply(lambda x: ZTModel(x.torque, x.thrust, x.bit_diameter, x.friction_coeff), axis=1)
#dataset['PFModel_ROP'] = dataset.apply(lambda x: PFModel(x.RPM, x.thrust, x.bit_diameter, x.friction_coeff), axis=1)

# MAKE MACHINE LEARNING MODEL____________________________________________________
X = dataset[['thrust', 'torque', 'RPM', 'bit_diameter', 'ZTModel_ROP']]
y = dataset.classification

train_X, test_X, train_Y, test_Y = train_test_split(X, y, train_size=0.7,shuffle=False, random_state=1)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=2000)

# Naïve Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='modified_huber', shuffle=True,random_state=101,tol=1e-3,max_iter=1000)

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=10,random_state=101,max_features=None,min_samples_leaf=5)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfm = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=1,random_state=101,max_features=None,min_samples_leaf=3)

# Support Vector Classifier
from sklearn.svm import SVC
svm = SVC(gamma='scale', C=1.0, random_state=101)

# Neural Network
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(solver='lbfgs',alpha=1e-5,max_iter=200, activation='relu',hidden_layer_sizes=(10,30,10),
                   random_state=1, shuffle=True)

# classification methods
m = [nb,lr,sgd,knn,dtree,rfm,svm,nn]
s = ['nb','lr','sgd','knn','dt','rfm','svm','nn']

# fit classifiers
print('Train Classifiers')
for i,x in enumerate(m):
    st = time.time()
    x.fit(train_X,train_Y)
    tf = str(round(time.time()-st,5))
    score = cross_val_score(x, X, y, cv = 5)
    print(s[i] + ' time: ' + tf + ', CV Score = ' + str(np.mean(score)))



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


# TODO: Pick Classification Model and Cross Validate?

print('Done')
