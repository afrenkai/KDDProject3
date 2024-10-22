import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging

def create_seq(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

# Preprocessing function
def preprocess(df: pd.DataFrame, engi: bool = False, seq: bool = False, seq_len=None, remove_outliers:bool=False):
    df = df.sort_values(by=['symbol', 'date']).reset_index(drop=True)
    scaler = MinMaxScaler()
    # Dropping unwanted columns and extracting features/target
    feat = df.drop(columns=['Unnamed: 0', 'symbol', 'date', 'DITM_IV'])

    if remove_outliers:
        print("Removing outliers using isolation forests")
        # use isolation forests to remove outliers
        pass

    if seq_len and seq:
        print("Creating sequences.")
        scaled_feat = scaler.fit_transform(feat)
        X, y = create_seq(scaled_feat, df['DITM_IV'].values, seq_len)
        print("Splitting data into training and testing sets.")
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=69, shuffle = False)
        return X_train, y_train, X_test, y_test
   
    X = feat
    y = df['DITM_IV']



    print("Splitting data into training and testing sets.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

    print("Standardizing training data.")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # If sequential data is required for LSTM, reshape the data into sequences
    # if seq and seq_len:
    #     print(f"Creating sequences with length {seq_len}.")
    #     X_train_seq, y_train_seq = create_seq(X_train_scaled, y_train, seq_len)
    #     X_test_seq, y_test_seq = create_seq(X_test_scaled, y_test, seq_len)
    #     return X_train_seq, y_train_seq, X_test_seq, y_test_seq


    return X_train_scaled, y_train, X_test_scaled, y_test

