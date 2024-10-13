import logging
from models import model_picker
import pandas as pd
from preprocess import preprocess
from harder_models import SGDRegressor
print('imports done')

logging.basicConfig(level=print, format='%(asctime)s - %(levelname)s - %(message)s')

df = pd.read_csv("../data/options.csv")
X_train_scaled, y_train, X_test_scaled, y_test = preprocess(df)
print('df done')

# SGD Linear Regression
print("Running SGD Linear Regression model...")
sgd_lr_results = model_picker('SGD Linear Regression',
                              X_train= X_train_scaled,
                              y_train= y_train,
                              X_test = X_test_scaled,
                              y_test = y_test)
print(f"SGD Linear Regression Results: {sgd_lr_results}")