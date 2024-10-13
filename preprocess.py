import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logging

def preprocess(df: pd.DataFrame, engi: bool = False):


    feat = ['strikes_spread', 'calls_contracts_traded', 'puts_contracts_traded',
        'calls_open_interest', 'puts_open_interest', 'expirations_number',
        'contracts_number', 'hv_20', 'hv_40', 'hv_60', 'hv_120', 'hv_180', 'VIX']
    X = df[feat]
    y = df['DITM_IV']
    scaler = MinMaxScaler()
    if engi:

    # we do a little feature engineering
        df['interaction_open_interest'] = df['calls_open_interest'] * df['puts_open_interest']
        df['calls_to_puts_traded_ratio'] = df['calls_contracts_traded'] / (df['puts_contracts_traded'] + 1)  # Adding 1 to avoid division by zero, went over it in class
        df['log_calls_traded'] = np.log1p(df['calls_contracts_traded'])  # log1p to handle zero values
        df['log_puts_traded'] = np.log1p(df['puts_contracts_traded']) # see comment above
        new_features = ['strikes_spread', 'calls_contracts_traded', 'puts_contracts_traded',
                    'calls_open_interest', 'puts_open_interest', 'expirations_number',
                    'contracts_number', 'hv_20', 'hv_40', 'hv_60', 'hv_120', 'hv_180', 'VIX',
                    'interaction_open_interest', 'calls_to_puts_traded_ratio', 'log_calls_traded', 'log_puts_traded']
        X_new = df[new_features]
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.2, random_state=69)
        X_train_new_scaled = scaler.fit_transform(X_train_new)
        X_test_new_scaled = scaler.transform(X_test_new)
        y_train_new = y_train_new.to_numpy()
        y_test_new = y_test_new.to_numpy
        return X_train_new_scaled, y_train_new, X_test_new_scaled, y_test_new

    logging.info("Splitting data into training and testing sets.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    scaler = MinMaxScaler()
    logging.info("Standardizing training data.")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, y_train, X_test_scaled, y_test







