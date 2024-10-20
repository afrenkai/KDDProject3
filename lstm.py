import logging
import pandas as pd
from preprocess import preprocess, create_seq
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for compatibility with SciKeras
import tensorflow as tf
from scikeras.wrappers import KerasRegressor  # Import from scikeras

print('Imports done')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess the dataset
df = pd.read_csv("data/sample.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['symbol', 'date']).reset_index(drop=True)
print('Dataframe processed')

# Scaling the features
scaler = MinMaxScaler()
sequence_length = 10
features = df.drop(columns=['Unnamed: 0', 'symbol', 'date', 'DITM_IV'])
target = df['DITM_IV']

scaled_features = scaler.fit_transform(features)
X, y = create_seq(scaled_features, target.values, sequence_length)


# Define the LSTM model creation function
def create_lstm_model(lstm_units_1=64, lstm_units_2=32, dense_units=16, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),  # Input layer
        tf.keras.layers.LSTM(lstm_units_1, return_sequences=True),  # First LSTM layer
        tf.keras.layers.LSTM(lstm_units_2),  # Second LSTM layer
        tf.keras.layers.Dense(dense_units, activation='relu'),  # Dense layer
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    # Compile the model with the given optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# Wrap the model using KerasRegressor from SciKeras
lstm_model = KerasRegressor(model=create_lstm_model, verbose=1)

# Define the parameter grid for grid search
param_grid = {
    'model__lstm_units_1': [32, 64, 128],  # First LSTM layer units
    'model__lstm_units_2': [16, 32, 64],  # Second LSTM layer units
    'model__dense_units': [8, 16, 32],  # Dense layer units
    'optimizer__learning_rate': [0.001, 0.0005],  # Learning rate
    'batch_size': [16, 32],  # Batch size
    'epochs': [10, 20]  # Number of epochs
}

# Create a 5-fold cross-validator
kfold = KFold(n_splits=5, shuffle=True)

# Define the GridSearchCV object
grid_search = GridSearchCV(estimator=lstm_model, param_grid=param_grid, cv=kfold, n_jobs=2,
                           scoring=make_scorer(mean_squared_error, greater_is_better=False))

# Fit the GridSearchCV
print("Starting Grid Search with 5-Fold Cross-Validation...")
grid_result = grid_search.fit(X, y)

# Summarize the results
print(f"Best score (MSE): {grid_result.best_score_}")
print(f"Best hyperparameters: {grid_result.best_params_}")

# Train the best model on the full dataset
best_model = grid_result.best_estimator_

# Evaluate on test set (if you have a separate test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
test_loss = best_model.score(X_test, y_test)
print(f"Best model test loss: {test_loss}")

# Make predictions with the best model
y_pred = best_model.predict(X_test)

# Compute evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}, R²: {r2}")
