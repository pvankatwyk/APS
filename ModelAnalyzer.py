def ModelAnalyzer(X,y):
    # TODO: Add more models and maybe add classification functionality
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    import warnings

    # Split dataset into train and test dataset (train_size is the proportion of train to test lengths)
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, train_size=0.7, shuffle=False, random_state=1)

    # Run several models and determine prediction accuracy using accuracy score.

    # Model Selection
    # Decision Tree
    start_dt = time.time()
    dt = DecisionTreeRegressor(random_state=1)
    dt.fit(train_X, train_Y)
    dt_test_predictions = dt.predict(test_X)
    dt_mae = mean_absolute_error(dt_test_predictions, test_Y)
    finish_dt = str(round(time.time() - start_dt, 5))
    out_dt = "Decision Tree MAE: " + str(dt_mae) + ', Time: ' + str(finish_dt) + ' seconds.'
    # print(out_dt)

    # Random Forest
    start_rf = time.time()
    rf = RandomForestRegressor(random_state=1, n_estimators=100)
    rf.fit(train_X, train_Y)
    rf_test_predictions = rf.predict(test_X)
    rf_mae = mean_absolute_error(rf_test_predictions, test_Y)
    finish_rf = str(round(time.time() - start_rf, 5))
    out_rf = "Random Forest MAE: " + str(rf_mae) + ', Time: ' + str(finish_rf) + ' seconds.'
    # print(out_rf)

    # Support Vector Regressor
    start_svr = time.time()
    svr = SVR(gamma='scale', C=1.0)
    svr.fit(train_X, train_Y)
    svr_test_predictions = svr.predict(test_X)
    svr_mae = mean_absolute_error(svr_test_predictions, test_Y)
    finish_svr = str(round(time.time() - start_svr, 5))
    out_svr = "Support Vector MAE: " + str(svr_mae) + ', Time: ' + str(finish_svr) + ' seconds.'
    # print(out_svr)


    # EXTRA TREES MODEL
    start_etr = time.time()
    etr = ExtraTreesRegressor(n_estimators=125)
    etr.fit(train_X, train_Y)
    etr_test_predictions = etr.predict(test_X)
    etr_mae = mean_absolute_error(etr_test_predictions, test_Y)
    finish_etr = str(round(time.time() - start_etr, 5))
    out_etr = "Extra Trees MAE: " + str(etr_mae) + ', Time: ' + str(finish_etr) + ' seconds.'
    # print(out_etr)

    out = out_dt + '\n' + out_rf + '\n' + out_svr + '\n' + out_etr
    return print(out)
