import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from harder_models import SGDRegressor, LSTM, DNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def model_picker(type: str, X_train, y_train, X_test, y_test):
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
            model = RandomForestRegressor()
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
            model = GradientBoostingRegressor()
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
            model = SVR()
            model.fit(X_train, y_train)
            logging.info("Training completed for Support Vector Regression.")

            y_hat = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_hat)
            r2 = r2_score(y_test, y_hat)
            mae = mean_absolute_error(y_test, y_hat)
            evs = explained_variance_score(y_test, y_hat)
            mse = rmse ** 2

        elif type == 'Long Short Term Memory':
            logging.info("Initializing Long Short Term Memory model...")
            X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            
            lstm_model = LSTM(input_size=X_train.shape[1], hidden_size=64, output_size=1, eta=0.001)
            lstm_model.train(X_train_lstm, y_train, epochs=100)
            logging.info("Training completed for LSTM model.")
            
            y_hat = lstm_model.forward(X_test_lstm)
            rmse = root_mean_squared_error(y_test, y_hat)
            r2 = r2_score(y_test, y_hat)
            mae = mean_absolute_error(y_test, y_hat)
            evs = explained_variance_score(y_test, y_hat)
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
            logging.info("Initializing Deep Neural Network model...")
            model = DNN(layer_sizes=[X_train.shape[1], 64, 32, 1], eta=0.01)
            model.train(X_train, y_train, epochs=100)
            logging.info("Training completed for DNN model.")

            y_hat, _ = model.forward(X_test)
            rmse = root_mean_squared_error(y_test, y_hat)
            r2 = r2_score(y_test, y_hat)
            mae = mean_absolute_error(y_test, y_hat)
            evs = explained_variance_score(y_test, y_hat)
            mse = rmse ** 2

        else:
            logging.error(f"Unsupported model type: {type}")
            return 'Not a supported model type or u buggin'

        logging.info(f"Evaluation results for {type}: RMSE: {rmse}, R2: {r2}, MAE: {mae}, EVS: {evs}, MSE: {mse}")
        return f'RMSE: {rmse}, R2: {r2}, MAE: {mae}, Explained Variance Score: {evs}, MSE: {mse}'

    except Exception as e:
        logging.exception(f"Error occurred while training {type}: {e}")
        return f"Error during training or evaluation of {type}: {str(e)}"
