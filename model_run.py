import logging
from models import model_picker
import pandas as pd
from tqdm import tqdm
from preprocess import preprocess
print('imports done')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tqdm.pandas()

df = pd.read_csv("data/options.csv")

print('df done')

logging.info("Splitting data into training and testing sets.")
X_train, X_test, y_train, y_test = preprocess(df)


# Linear Regression
# logging.info("Running Linear Regression model...")
# ols_results = model_picker('OLS Linear Regression', 
#                            X_train=X_train_scaled, 
#                            y_train=y_train, 
#                            X_test=X_test_scaled, 
#                            y_test=y_test)
# logging.info(f"OLS Results: {ols_results}")

# # Gradient Boosting Regression
# logging.info("Running Gradient Boost Regression model...")
# gb_results = model_picker('Gradient Boost Regression', 
#                           X_train=X_train_scaled, 
#                           y_train=y_train, 
#                           X_test=X_test_scaled, 
#                           y_test=y_test)
# logging.info(f"Gradient Boosting Results: {gb_results}")

# # Support Vector Regression
# logging.info("Running Support Vector Regression model...")
# svr_results = model_picker('Support Vector Regression', 
#                            X_train=X_train_scaled, 
#                            y_train=y_train, 
#                            X_test=X_test_scaled, 
#                            y_test=y_test)
# logging.info(f"SVR Results: {svr_results}")

# LSTM
# logging.info("Running Long Short Term Memory model...")
# lstm_results = model_picker('Long Short Term Memory', 
#                             X_train=X_train_scaled, 
#                             y_train=y_train, 
#                             X_test=X_test_scaled, 
#                             y_test=y_test)
# logging.info(f"LSTM Results: {lstm_results}")

# SGD Linear Regression
logging.info("Running SGD Linear Regression model...")
sgd_lr_results = model_picker('SGD Linear Regression', 
                              X_train= X_train, 
                              y_train= y_train, 
                              X_test = X_test, 
                              y_test = y_test)
logging.info(f"SGD Linear Regression Results: {sgd_lr_results}")

# # Deep Neural Network
# logging.info("Running Deep Neural Network model...")
# dnn_results = model_picker('Deep Neural Network',  
#                            X_train=X_train_scaled, 
#                            y_train=y_train, 
#                            X_test=X_test_scaled, 
#                            y_test=y_test)
# logging.info(f"DNN Results: {dnn_results}")
