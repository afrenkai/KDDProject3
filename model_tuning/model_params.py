import numpy as np
# RF
RF_PARAMS = {
    'n_estimators': [200,300],
    'max_depth': [20, 30, 40],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'n_jobs' : [-1],
    'random_state': [69],
}

RF_NAME = "Random Forest"

# GB
HGB_PARAMS = {
        'max_depth': [None, 30, 40],
        'max_features': list(np.linspace(0.5,1,3)),
        'max_iter': [200, 300, 400],
        'min_samples_leaf': [30, 40],
        'learning_rate' : [0.1, 0.01, 0.001],
        'random_state': [69]
}

HGB_NAME = "Hist Gradient Boosting"

# no tuning for OLS
OLS_NAME = "OLS"

# ElasticNet 
EN_NAME = "Elastic Net"
# l1_ratio = 0 == L2 Pen, l1_ratio = 1 == L1 Pen
EN_PARAMS = {
        'l1_ratio': list(np.linspace(0.2,1,6)),
        'random_state': [69]
}
<<<<<<< HEAD

SVR_NAME = "Support Vector"



# DNN

DNN_PARAMS = {
    'epochs': [10,20,50],
    'batch_size': [16,32],
    'learning_rate': [0.001, 0.005, 0.01]
}

DNN_NAME = "Deep Neural Network"
=======
>>>>>>> 257a803f804b9b0c7f0a99db232359e6c7dae415
