import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging


# def create_seq(data, target_data, seq_len=10):
#     x, y = [], []
#     for i in range(len(data) - seq_len):
#         x.append(data[i:i+seq_len])
#         if i + seq_len < len(target_data):
#             y.append(target_data.iloc[i + seq_len])
#     return np.array(x), np.array(y)


# def create_data_loader(X, y, batch_size: int=32):
#     X_tensor = torch.tensor(X, dtype=torch.float32)
#     y_tensor = torch.tensor(y, dtype = torch.float32)
#     ds = TensorDataset(X_tensor, y_tensor)
#     return DataLoader(ds, batch_size=batch_size, shuffle = True)

def preprocess(df: pd.DataFrame, engi: bool = False, seq: bool = False, seq_len=None):
    scaler = MinMaxScaler()

    feat = ['strikes_spread', 'calls_contracts_traded', 'puts_contracts_traded',
            'calls_open_interest', 'puts_open_interest', 'expirations_number',
            'contracts_number', 'hv_20', 'hv_40', 'hv_60', 'hv_75', 'hv_90',
            'hv_120', 'hv_180', 'hv_200', 'VIX']

    X = df[feat]
    y = df['DITM_IV']

    print("Splitting data into training and testing sets.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

    print("Standardizing training data.")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # # If seq data is required (LSTM), reshape data into seq
    # if seq and seq_len:
    #     print(f"Creating sequences with length {seq_len}.")
    #     X_train_seq, y_train_seq = create_seq(X_train_scaled, y_train, seq_len)
    #     X_test_seq, y_test_seq = create_seq(X_test_scaled, y_test, seq_len)
    #     return X_train_seq, y_train_seq, X_test_seq, y_test_seq

    return X_train_scaled, y_train, X_test_scaled, y_test
