import logging
from models import model_picker
import pandas as pd
# from tqdm import tqdm
# from preprocess import preprocess

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
print('imports done')

logging.basicConfig(level=print, format='%(asctime)s - %(levelname)s - %(message)s')


df = pd.read_csv("data/options.csv")

print('df done')

feat = ['strikes_spread', 'calls_contracts_traded', 'puts_contracts_traded',
        'calls_open_interest', 'puts_open_interest', 'expirations_number',
        'contracts_number', 'hv_20', 'hv_40', 'hv_60', 'hv_120', 'hv_180', 'VIX']
X = df[feat]
y = df['DITM_IV']

print("Splitting data into training and testing sets.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
scaler = MinMaxScaler()
print("Standardizing training data.")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# print("Splitting data into training and testing sets.")
# X_train, X_test, y_train, y_test = preprocess(df)

#
# Linear Regression
print("Running Linear Regression model...")
ols_results = model_picker('OLS Linear Regression',
                           X_train = X_train_scaled,
                           y_train=y_train,
                           X_test=X_test_scaled,
                           y_test=y_test)
print(f"OLS Results: {ols_results}")

#Random Forest
print('Running Random Forest Regression model')
rf_results = model_picker('Random Forest Regression',
                          X_train=X_train_scaled,
                          y_train=y_train,
                          X_test=X_test_scaled,
                          y_test=y_test)
print(f'Random Forest Results: {rf_results}')

# Gradient Boosting Regression
# print("Running Gradient Boost Regression model...")
# gb_results = model_picker('Gradient Boost Regression',
#                           X_train=X_train_scaled,
#                           y_train=y_train,
#                           X_test=X_test_scaled,
#                           y_test=y_test)
# print(f"Gradient Boosting Results: {gb_results}")

# Support Vector Regression
print("Running Support Vector Regression model...")
svr_results = model_picker('Support Vector Regression',
                           X_train=X_train_scaled,
                           y_train=y_train,
                           X_test=X_test_scaled,
                           y_test=y_test)
print(f"SVR Results: {svr_results}")

# LSTM
# print("Running Long Short Term Memory model...")
# lstm_results = model_picker('Long Short Term Memory', 
#                             X_train=X_train_scaled, 
#                             y_train=y_train, 
#                             X_test=X_test_scaled, 
#                             y_test=y_test)
# print(f"LSTM Results: {lstm_results}")

# SGD Linear Regression
# print("Running SGD Linear Regression model...")
# sgd_lr_results = model_picker('SGD Linear Regression',
#                               X_train= X_train,
#                               y_train= y_train,
#                               X_test = X_test,
#                               y_test = y_test)
# print(f"SGD Linear Regression Results: {sgd_lr_results}")

# # Deep Neural Network
# print("Running Deep Neural Network model...")
# dnn_results = model_picker('Deep Neural Network',
#                            X_train=X_train,
#                            y_train=y_train,
#                            X_test=X_test,
#                            y_test=y_test)
# print(f"DNN Results: {dnn_results}")
