import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from deep_models import SGDRegressor, OptionsLSTM, OptionsNN
import torch
import tensorflow as tf
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def model_picker(type: str, X_train, y_train, X_test, y_test, init_type = None, seq_len = None):
    logging.info(f"Starting model picker for model: {type}")

    if X_train is None or y_train is None or X_test is None or y_test is None:
        logging.error("Error: Missing data in input sets.")
        return "Error: sm missin vro"

    model = None
    rmse = None
    mae = None
    r2 = None
    evs = None
    mse = None

    try:
        if type == 'OLS Linear Regression':
            logging.info("Initializing OLS Linear Regression model...")
            model = LinearRegression()
            model.fit(X_train, y_train)
            logging.info("Training completed for OLS Linear Regression.")
            
            y_hat = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_hat)
            r2 = r2_score(y_test, y_hat)
            mae = mean_absolute_error(y_test, y_hat)
            evs = explained_variance_score(y_test, y_hat)
            mse = rmse ** 2

        elif type == 'Random Forest Regression':
            logging.info("Initializing Random Forest Regression model...")
            model = RandomForestRegressor(verbose = 1)
            model.fit(X_train, y_train)
            logging.info("Training completed for Random Forest Regression.")

            y_hat = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_hat)
            r2 = r2_score(y_test, y_hat)
            mae = mean_absolute_error(y_test, y_hat)
            evs = explained_variance_score(y_test, y_hat)
            mse = rmse ** 2

        elif type == 'Gradient Boost Regression':
            logging.info("Initializing Gradient Boost Regression model...")
            model = GradientBoostingRegressor(verbose = 1)
            model.fit(X_train, y_train)
            logging.info("Training completed for Gradient Boost Regression.")

            y_hat = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_hat)
            r2 = r2_score(y_test, y_hat)
            mae = mean_absolute_error(y_test, y_hat)
            evs = explained_variance_score(y_test, y_hat)
            mse = rmse ** 2

        elif type == 'Support Vector Regression':
            logging.info("Initializing Support Vector Regression model...")
            model = SVR(verbose = 1)
            model.fit(X_train, y_train)
            logging.info("Training completed for Support Vector Regression.")

            y_hat = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_hat)
            r2 = r2_score(y_test, y_hat)
            mae = mean_absolute_error(y_test, y_hat)
            evs = explained_variance_score(y_test, y_hat)
            mse = rmse ** 2

        elif type == 'Long Short Term Memory':
            # Reshape for LSTM model, assuming data is already scaled
            X_train_lstm = X_train.reshape((X_train.shape[0], seq_len, X_train.shape[1]))
            X_test_lstm = X_test.reshape((X_test.shape[0], seq_len, X_test.shape[1]))

            # Convert to TensorFlow tensors
            X_train_lstm = tf.convert_to_tensor(X_train_lstm, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
            X_test_lstm = tf.convert_to_tensor(X_test_lstm, dtype=tf.float32)
            y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

            # Initialize and train the LSTM model
            input_size = X_train_lstm.shape[2]  # Number of features
            lstm_model = OptionsLSTM(input_size=input_size, hidden_size=64, num_layers=2, eta=0.001)
            lstm_model.train_model(X_train_lstm, y_train, epochs=10, batch_size=32)

            # Evaluate the LSTM model
            test_loss = lstm_model.evaluate(X_test_lstm, y_test)
            logging.info(f'Test Loss (MSE): {test_loss:.4f}')

            # Get predictions
            y_hat = lstm_model.predict(X_test_lstm).flatten()

            # Calculate metrics
            rmse = root_mean_squared_error(y_test.numpy(), y_hat)
            r2 = r2_score(y_test.numpy(), y_hat)
            mae = mean_absolute_error(y_test.numpy(), y_hat)
            evs = explained_variance_score(y_test.numpy(), y_hat)
            mse = rmse ** 2

        elif type == 'SGD Linear Regression':
            logging.info("Initializing SGD Linear Regression model...")
            model = SGDRegressor()
            model.fit(X_train, y_train)
            logging.info("Training completed for SGD Linear Regression.")

            y_hat = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_hat)
            r2 = r2_score(y_test, y_hat)
            mae = mean_absolute_error(y_test, y_hat)
            evs = explained_variance_score(y_test, y_hat)
            mse = rmse ** 2

        elif type == 'Deep Neural Network':
            input_size = X_train.shape[1]
            model = OptionsNN(input_size = input_size, init_type = init_type)

            if not isinstance(X_train, torch.Tensor):
                X_train = torch.tensor(X_train, dtype = torch.float32)
                y_train = torch.tensor(y_train, dtype = torch.float32).reshape(-1, 1)
                X_test = torch.tensor(X_test, dtype = torch.float32)
                y_test = torch.tensor(y_test.to_numpy(), dtype = torch.float32).reshape(-1, 1)
            model.train_model(X_train, y_train)
            logging.info('training finished for DNN')
            y_hat_tensor = model.predict(X_test)
            model.save_model('weights.pth')
            y_hat = y_hat_tensor.detach().cpu().numpy()
            y_test_np = y_test.detach().cpu().numpy()

            rmse = root_mean_squared_error(y_test_np, y_hat)
            r2 = r2_score(y_test_np, y_hat)
            mae = mean_absolute_error(y_test_np, y_hat)
            evs = explained_variance_score(y_test_np, y_hat)
            mse = rmse ** 2  # Mean squared error

        else:
            logging.error(f"Unsupported model type: {type}")
            return 'Not a supported model type or u buggin'

        logging.info(f"Evaluation results for {type}: RMSE: {rmse}, R2: {r2}, MAE: {mae}, EVS: {evs}, MSE: {mse}")
        return f'RMSE: {rmse}, R2: {r2}, MAE: {mae}, Explained Variance Score: {evs}, MSE: {mse}'

    except Exception as e:
        logging.exception(f"Error occurred while training {type}: {e}")
        return f"Error during training or evaluation of {type}: {str(e)}"
