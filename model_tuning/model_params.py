# RF
RF_PARAMS = {
    'n_estimators': [100, 200,300],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'n_jobs' : [1],
    'random_state': [69]
}

RF_NAME = "Random Forest"

# GB
GB_PARAMS = {
        'n_estimators': [200,300],
        'max_depth': [3, 10, 20],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'learning_rate' : [0.1, 0.01, 0.001],
        'random_state': [69]
}

GB_NAME = "Gradient Boosting"

# no tuning for OLS
OLS_NAME = "OLS"

# SVR
SVR_PARAMS = {
        'kernel': ['linear','poly', 'rbf'],
        'degree': [2,3,4],
        'C': [0.5, 0.8, 1],
}

SVR_NAME = "Support Vector"



# DNN

DNN_PARAMS = {
    'epochs': [10,20,50],
    'batch_size': [16,32],
    'learning_rate': [0.001, 0.005, 0.01]
}

DNN_NAME = "Deep Neural Network"