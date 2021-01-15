# Compilation of subfunctions used in data cleaning, processing, etc.

def PrepareData(folder, profiling):
    # INPUTS:
    #   - folder: (str) Filepath of the folder containing the data
    #   - profiling: (bool) Whether a profiling report will be generated
    # OUTPUTS:
    #   - data: Returns the dataframe to be analyzed
    # (Original code taken from visualize_all.py - Hedengren 11/13)
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import math
    from glob import glob
    import warnings
    warnings.filterwarnings('ignore')
    #sns.set_theme()

    # Import data
    d = folder                                                  # data directory
    p = './profiling/'                                          # profiling directory
    os.chdir(d)
    files = glob('*')
    os.chdir('../')
    flist = [f[0:-4] for f in files]                            # Gets rid of ".csv" at the end

    # Compile all data
    dataframes = []
    for f in flist:
        data = pd.read_csv(d+f+'.csv',quotechar='"')            # read with quotechar
        data = data.iloc[::-1].reset_index()                    # reverse order
        data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])   # to datetime
        n = len(data)                                           # add ROP
        x = np.zeros(n)*np.nan
        y = [True]*n
        # Calculate time stamp differences
        for i in range(1,n):
            x[i] = (data['TimeStamp'][i]-data['TimeStamp'][i-1]).total_seconds()
            y[i] = True if data['Rod Count'][i]>data['Rod Count'][i-1] else False
        x[0] = np.mean(x[1:3]) # average of 2 and 3 for 1st time point only
        data['ROP (ft/min)'] = np.array(y)*60.0*10.0/x
        data['deltaTime'] = x
        data['Drill'] = y
        data = data[data['Drill']]                              # Remove all but forward drill
        #data = data[data['ROP (ft/min)']<4]                    # Keep data with ROP < 4
        data['Time'] = data['TimeStamp']                        # Time last
        del data['TimeStamp']                                   # remove TimeStamp
        del data['index']                                       # remove index
        #data['Location'] = str(f)                               # Create Location Column with file name
        dataframes.append(data)

    # combined = np.vstack(dataframes)
    # data = pd.DataFrame(dataframes,columns=data.columns)
    data = pd.concat(dataframes)                                # Combine all dataframes from each dataset

    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["ROP (ft/min)"], how="all") # Drops rows with infinite ROP (no time change)
    data = data.reset_index(drop = True)
    data['ROP (ft/hr)'] = data['ROP (ft/min)'] * 60

    # MSE Calculation + Plot -------------------------------------------------
    #bit_diameter = 4.0
    #Area = math.pi*(bit_diameter/2.0)**2.0
    #data['MSE'] = (data['Thrust Force Max (lbf)']/Area)+((2*math.pi*data['Rotation Speed Max (rpm)']*data['Rotation Torque Max (ft-lb)'])/(Area*data['ROP (ft/hr)']))

    # If you want a profile report..
    if profiling:
        from pandas_profiling import ProfileReport
        f = 'combined'
        file = p + f + '.html'
        if not os.path.isfile(file):
            prof = ProfileReport(data)
            prof.to_file(output_file=file)
        else:
            print('Delete file '+file+' to regenerate profiling report')

    return data



def ModelAnalyzer(X,y, regressor = True):
    # INPUTS:
    #   - X: (DataFrame) Explanatory variables to be used as features for ML models
    #   - y: (Vector) Response variables to be used as target for ML models
    #   - regressor: (bool) Determines whether a regressor or classifier will be used
    # OUTPUTS:
    #   - out: (str) Multiline report of the accuracy and fit time of each model

    import time
    from sklearn.metrics import mean_absolute_error, accuracy_score
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings('ignore')

    # Split dataset into train and test dataset (train_size is the proportion of train to test lengths)
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, train_size=0.5, shuffle=False, random_state=1)

    if regressor:
        # Run several models and determine prediction accuracy using accuracy score.

        # Model Selection
        # Decision Tree
        from sklearn.tree import DecisionTreeRegressor
        start_dt = time.time()
        dt = DecisionTreeRegressor(random_state=1)
        dt.fit(train_X, train_Y)
        dt_test_predictions = dt.predict(test_X)
        dt_mae = mean_absolute_error(dt_test_predictions, test_Y)
        finish_dt = str(round(time.time() - start_dt, 5))
        out_dt = "Decision Tree MAE: " + str(dt_mae) + ', Time: ' + str(finish_dt) + ' seconds.'

        # Random Forest
        from sklearn.ensemble import RandomForestRegressor
        start_rf = time.time()
        rf = RandomForestRegressor(random_state=1, max_features= 'auto', min_samples_split=2, min_samples_leaf=1, n_estimators = 650)
        rf.fit(train_X, train_Y)
        rf_test_predictions = rf.predict(test_X)
        rf_mae = mean_absolute_error(rf_test_predictions, test_Y)
        finish_rf = str(round(time.time() - start_rf, 5))
        out_rf = "Random Forest MAE: " + str(rf_mae) + ', Time: ' + str(finish_rf) + ' seconds.'

        # Support Vector Regressor
        from sklearn.svm import SVR
        start_svr = time.time()
        svr = SVR(gamma='scale', C=1.0)
        svr.fit(train_X, train_Y)
        svr_test_predictions = svr.predict(test_X)
        svr_mae = mean_absolute_error(svr_test_predictions, test_Y)
        finish_svr = str(round(time.time() - start_svr, 5))
        out_svr = "Support Vector MAE: " + str(svr_mae) + ', Time: ' + str(finish_svr) + ' seconds.'

        # EXTRA TREES MODEL
        from sklearn.ensemble import ExtraTreesRegressor
        start_etr = time.time()
        etr = ExtraTreesRegressor(max_features='auto', n_estimators=125, min_samples_split=3, random_state=1)
        etr.fit(train_X, train_Y)
        etr_test_predictions = etr.predict(test_X)
        etr_mae = mean_absolute_error(etr_test_predictions, test_Y)
        finish_etr = str(round(time.time() - start_etr, 5))
        out_etr = "Extra Trees MAE: " + str(etr_mae) + ', Time: ' + str(finish_etr) + ' seconds.'

        from sklearn.linear_model import LassoCV
        start_lasso = time.time()
        lasso = LassoCV()
        lasso.fit(train_X, train_Y)
        lasso_test_predictions = lasso.predict(test_X)
        lasso_mae = mean_absolute_error(lasso_test_predictions, test_Y)
        finish_lasso = str(round(time.time() - start_lasso, 5))
        out_lasso = "Lasso MAE: " + str(lasso_mae) + ', Time: ' + str(finish_lasso) + ' seconds.'

        from sklearn.linear_model import RidgeCV
        start_ridge = time.time()
        ridge = RidgeCV()
        ridge.fit(train_X, train_Y)
        ridge_test_predictions = ridge.predict(test_X)
        ridge_mae = mean_absolute_error(ridge_test_predictions, test_Y)
        finish_ridge = str(round(time.time() - start_ridge, 5))
        out_ridge = "Ridge MAE: " + str(ridge_mae) + ', Time: ' + str(finish_ridge) + ' seconds.'

        from sklearn.linear_model import ElasticNetCV
        start_en = time.time()
        en = ElasticNetCV()
        en.fit(train_X, train_Y)
        en_test_predictions = en.predict(test_X)
        en_mae = mean_absolute_error(en_test_predictions, test_Y)
        finish_en = str(round(time.time() - start_en, 5))
        out_en = "Elastic Net MAE: " + str(en_mae) + ', Time: ' + str(finish_en) + ' seconds.'

        out = out_dt + '\n' + out_rf + '\n' + out_svr + '\n' + out_etr + '\n' + out_lasso + '\n' + out_ridge + '\n' + out_en

    else:
        # Run several models and determine prediction accuracy using accuracy score.

        # Logistic Regression
        from sklearn.linear_model import LogisticRegression
        start = time.time()
        lr = LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=2000)
        lr.fit(train_X, train_Y)
        lr_predictions = lr.predict(test_X)
        finish_lr = str(round(time.time()-start,5))
        lr_accuracy = accuracy_score(test_Y, lr_predictions)
        out_lr = "Logistic Regression Accuracy: " + str(lr_accuracy) + ', Time: ' + str(finish_lr) + ' seconds.'

        # Naïve Bayes
        from sklearn.naive_bayes import GaussianNB
        start = time.time()
        nb = GaussianNB()
        nb.fit(train_X, train_Y)
        nb_predictions = nb.predict(test_X)
        finish_nb = str(round(time.time()-start,5))
        nb_accuracy = accuracy_score(test_Y, nb_predictions)
        out_nb = "Naive Bayes Accuracy: " + str(nb_accuracy) + ', Time: ' + str(finish_nb) + ' seconds.'

        # Stochastic Gradient Descent
        from sklearn.linear_model import SGDClassifier
        start = time.time()
        sgd = SGDClassifier(loss='modified_huber', shuffle=True,random_state=101,tol=1e-3,max_iter=1000)
        sgd.fit(train_X, train_Y)
        sgd_predictions = sgd.predict(test_X)
        finish_sgd = str(round(time.time()-start,5))
        sgd_accuracy = accuracy_score(test_Y, sgd_predictions)
        out_sgd = "SGD Accuracy: " + str(sgd_accuracy) + ', Time: ' + str(finish_sgd) + ' seconds.'

        # K-Nearest Neighbors
        from sklearn.neighbors import KNeighborsClassifier
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(train_X, train_Y)
        knn_predictions = knn.predict(test_X)
        finish_knn = str(round(time.time()-start,5))
        knn_accuracy = accuracy_score(test_Y, knn_predictions)
        out_knn = "KNN Accuracy: " + str(knn_accuracy) + ', Time: ' + str(finish_knn) + ' seconds.'

        # Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        start = time.time()
        dt = DecisionTreeClassifier(max_depth=10,random_state=101,max_features=None,min_samples_leaf=5)
        dt.fit(train_X, train_Y)
        dt_predictions = dt.predict(test_X)
        finish_dt = str(round(time.time()-start,5))
        dt_accuracy = accuracy_score(test_Y, dt_predictions)
        out_dt = "Decision Tree Accuracy: " + str(dt_accuracy) + ', Time: ' + str(finish_dt) + ' seconds.'

        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        start = time.time()
        rfm = RandomForestClassifier(n_estimators=125, oob_score=True, n_jobs=1, random_state=101, max_features=None,
                                     min_samples_leaf=3)
        rfm.fit(train_X, train_Y)
        rfm_predictions = rfm.predict(test_X)
        finish_rfm = str(round(time.time() - start, 5))
        rfm_accuracy = accuracy_score(test_Y, rfm_predictions)
        out_rfm = "Random Forest Accuracy: " + str(rfm_accuracy) + ', Time: ' + str(finish_rfm) + ' seconds.'

        # Support Vector Classifier
        from sklearn.svm import SVC
        start = time.time()
        svm = SVC(gamma='scale', C=1.0, random_state=101)
        svm.fit(train_X, train_Y)
        svm_predictions = svm.predict(test_X)
        finish_svm = str(round(time.time()-start,5))
        svm_accuracy = accuracy_score(test_Y, svm_predictions)
        out_svm = "SVC Accuracy: " + str(svm_accuracy) + ', Time: ' + str(finish_svm) + ' seconds.'

        # Extra Trees
        from sklearn.ensemble import ExtraTreesClassifier
        start = time.time()
        etc = ExtraTreesClassifier(n_estimators=125)
        etc.fit(train_X, train_Y)
        etc_predictions = etc.predict(test_X)
        finish_etc = str(round(time.time()-start, 5))
        etc_accuracy = accuracy_score(test_Y, etc_predictions)
        out_etc = "Extra Trees Accuracy: " + str(etc_accuracy) + ', Time: ' + str(finish_etc) + ' seconds.'

        out = out_lr + '\n' + out_nb + '\n' + out_sgd + '\n' + out_knn + '\n' + out_dt + '\n' + out_rfm + '\n' + out_svm +'\n' + out_etc

    return print(out)


def ParamOptimize(parameters, model, train_X, train_Y):
    # INPUTS:
    #   - parameters: (dict) Parameters to be tested/optimized
    #   - model: (ML) Machine Learning model to be optimized
    #   - train_X: (DataFrame) Training features
    #   - train_Y: (Vector) Training target
    # OUTPUTS:
    #   - gsc.best_estimator: (str) Prints optimized parameters

    # parameters = {
    #     'n_estimators': range(25,200,1)
    #     #'min_samples_leaf': [1, 2, 3],
    #     #'min_samples_split': [2,3,4]
    # }


    from sklearn.model_selection import GridSearchCV
    gsc = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    gsc.fit(train_X, train_Y)
    return print(gsc.best_params_)

def Quartile_Analysis(data):
    import pandas as pd
    import numpy as np
    # Calculate quartiles boundaries
    quartiles = np.array(data.rop.quantile([0.25, 0.5, 0.75]))
    quartile1 = quartiles[2]
    quartile2 = quartiles[1]
    quartile3 = quartiles[0]

    quartile1_dataset = data.loc[data.rop >= quartile1]
    quartile2_dataset = data.loc[(data.rop < quartile1) & (data.rop >= quartile2)]
    quartile3_dataset = data.loc[(data.rop < quartile2) & (data.rop >= quartile3)]
    quartile4_dataset = data.loc[(data.rop < quartile3) & (data.rop >= min(data.rop))]

    # Make "summary" dataset to view the ROP characteristics of each quartile
    rpm = np.array([np.mean(quartile1_dataset.rpm), np.mean(quartile2_dataset.rpm), np.mean(quartile3_dataset.rpm),
                    np.mean(quartile4_dataset.rpm)])
    torque = np.array(
        [np.mean(quartile1_dataset.torque), np.mean(quartile2_dataset.torque), np.mean(quartile3_dataset.torque),
         np.mean(quartile4_dataset.torque)])
    thrust = np.array(
        [np.mean(quartile1_dataset.thrust), np.mean(quartile2_dataset.thrust), np.mean(quartile3_dataset.thrust),
         np.mean(quartile4_dataset.thrust)])
    mud_flow = np.array(
        [np.mean(quartile1_dataset.mud_flow), np.mean(quartile2_dataset.mud_flow), np.mean(quartile3_dataset.mud_flow),
         np.mean(quartile4_dataset.mud_flow)])
    mud_press = np.array(
        [np.mean(quartile1_dataset.mud_press), np.mean(quartile2_dataset.mud_press), np.mean(quartile3_dataset.mud_press),
         np.mean(quartile4_dataset.mud_press)])
    thrust_speed = np.array([np.mean(quartile1_dataset.thrust_speed), np.mean(quartile2_dataset.thrust_speed),
                             np.mean(quartile3_dataset.thrust_speed), np.mean(quartile4_dataset.thrust_speed)])
    pull_force = np.array([np.mean(quartile1_dataset.pull_force), np.mean(quartile2_dataset.pull_force),
                           np.mean(quartile3_dataset.pull_force), np.mean(quartile4_dataset.pull_force)])
    pull_speed = np.array([np.mean(quartile1_dataset.pull_speed), np.mean(quartile2_dataset.pull_speed),
                           np.mean(quartile3_dataset.pull_speed), np.mean(quartile4_dataset.pull_speed)])
    ds_length = np.array(
        [np.mean(quartile1_dataset.ds_length), np.mean(quartile2_dataset.ds_length), np.mean(quartile3_dataset.ds_length),
         np.mean(quartile4_dataset.ds_length)])
    rop = np.array([np.mean(quartile1_dataset.rop), np.mean(quartile2_dataset.rop), np.mean(quartile3_dataset.rop),
                    np.mean(quartile4_dataset.rop)])

    summary = pd.DataFrame([rpm, torque, thrust, mud_flow, mud_press, thrust_speed, pull_force, pull_speed, ds_length, rop])
    summary = summary.transpose()
    summary.columns = ['rpm', 'torque', 'thrust', 'mud_flow', 'mud_press', 'thrust_speed', 'pull_force', 'pull_speed',
                       'ds_length', 'rop']
    summary.index = ['quartile_1', 'quartile_2', 'quartile_3', 'quartile_4']

    # To view "summary" DF, either view in debug mode or uncomment the following two lines and specify the output csv path
    out_path = r'C:/Users/Peter/Downloads/ROP_Summary.csv'
    #summary.to_csv(out_path)

    # Calculate median ROP value for entire dataset for comparison
    median_rop = np.median(data.rop)

    # Print percent change
    # Note: I changed "increase" and "decrease" in the results and used absolute value for negative changes.
    q1_print = 'By keeping the drilling parameters within the first quartile ROP, we INCREASE the average speed by ' + str(
        round((summary.rop[0] / median_rop - 1.00) * 100.00, 2)) + '%.'
    q2_print = 'By keeping the drilling parameters within the second quartile ROP, we INCREASE the average speed by ' + str(
        round((summary.rop[1] / median_rop - 1.00) * 100.00, 2)) + '%.'
    q3_print = 'By keeping the drilling parameters within the third quartile ROP, we DECREASE the average speed by ' + str(
        abs(round((summary.rop[2] / median_rop - 1.00) * 100.00, 2))) + '%.'
    q4_print = 'By keeping the drilling parameters within the fourth quartile ROP, we DECREASE the average speed by ' + str(
        abs(round((summary.rop[3] / median_rop - 1.00) * 100.00, 2))) + '%.'
    print(q1_print)
    print(q2_print)
    print(q3_print)
    print(q4_print)
