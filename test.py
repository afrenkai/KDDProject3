import logging
from models import model_picker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
print('imports done')


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tqdm.pandas()

df = pd.read_csv("hf://datasets/gauss314/options-IV-SP500/data_IV_USA.csv")

print('df done')
# Select features and target
feat = ['strikes_spread', 'calls_contracts_traded', 'puts_contracts_traded', 
        'calls_open_interest', 'puts_open_interest', 'expirations_number', 
        'contracts_number', 'hv_20', 'hv_40', 'hv_60', 'hv_120', 'hv_180', 'VIX']
X = df[feat]
y = df['DITM_IV']

# Split data
logging.info("Splitting data into training and testing sets.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Standardize the data
scaler = StandardScaler()
logging.info("Standardizing training data.")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Picker with logging for various models

# Linear Regression
logging.info("Running Linear Regression model...")
ols_results = model_picker('OLS Linear Regression', 
                           X_train=X_train_scaled, 
                           y_train=y_train, 
                           X_test=X_test_scaled, 
                           y_test=y_test)
logging.info(f"OLS Results: {ols_results}")

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
                              X_train=X_train_scaled, 
                              y_train=y_train, 
                              X_test=X_test_scaled, 
                              y_test=y_test)
logging.info(f"SGD Linear Regression Results: {sgd_lr_results}")

# # Deep Neural Network
# logging.info("Running Deep Neural Network model...")
# dnn_results = model_picker('Deep Neural Network',  
#                            X_train=X_train_scaled, 
#                            y_train=y_train, 
#                            X_test=X_test_scaled, 
#                            y_test=y_test)
# logging.info(f"DNN Results: {dnn_results}")
