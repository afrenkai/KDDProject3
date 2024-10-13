import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logging

def preprocess(df: pd.DataFrame, engi: bool = False):


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
    return X_train_scaled, y_train, X_test_scaled, y_test








