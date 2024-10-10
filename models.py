from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from harder_models import SGDRegressor, LSTM, DNN
from tqdm import tqdm

def model_picker(type: str, X_train, y_train, X_test, y_test):
    if X_train is None or y_train is None or X_test is None or y_test is None:
        return "Error: sm missin vro"
    model = None
    rmse = None
    mae = None
    r2 = None
    evs = None
    mse = None

    macro = f' RMSE: {rmse}, R2: {r2}, MAE: {mae}, Explained Variance Score: {evs}, MSE: {mse}'
    if type == 'OLS Linear Regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_hat)
        r2 = r2_score(y_test, y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        evs = explained_variance_score(y_test, y_hat)
        mse = rmse ** 2
        return macro
    elif type == 'Random Forest Regression':
        model = RandomForestRegressor(n_estimators=100, n_jobs=1)
        with tqdm(total=model.n_estimators, desc="Training Random Forest", unit="tree") as pbar:
            for i in range(model.n_estimators):
                model.fit(X_train, y_train)
                pbar.update(1)
        y_hat = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_hat)
        r2 = r2_score(y_test, y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        evs = explained_variance_score(y_test, y_hat)
        mse = rmse ** 2
        return macro
    elif type == 'Gradient Boost Regression':
        model = GradientBoostingRegressor(X_train, y_train)
        y_hat = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_hat)
        r2 = r2_score(y_test, y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        evs = explained_variance_score(y_test, y_hat)
        mse = rmse ** 2
        return macro
    elif type == 'Support Vector Regression':
        model = SVR(X_train, y_train)
        y_hat = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_hat)
        r2 = r2_score(y_test, y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        evs = explained_variance_score(y_test, y_hat)
        mse = rmse ** 2
        return macro
    elif type == 'Long Short Term Memory':
        # shape : [sample, timestep, feature]
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        lstm_model = LSTM(input_size=X_train.shape[1], hidden_size=64, output_size=1, eta=0.001)
        lstm_model.train(X_train_lstm, y_train, epochs=100)
        y_hat = lstm_model.forward(X_test_lstm)
        rmse = root_mean_squared_error(y_test, y_hat)
        r2 = r2_score(y_test, y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        evs = explained_variance_score(y_test, y_hat)
        mse = rmse ** 2
        return macro
    elif type == 'SGD Linear Regression':
        model = SGDRegressor()
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_hat)
        r2 = r2_score(y_test, y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        evs = explained_variance_score(y_test, y_hat)
        mse = model.mse(y_test, y_hat)
        return macro
    elif type == 'Deep Neural Network':
        model = DNN(layer_sizes=[X_train.shape[0], 64, 32, 1], eta=0.01)
        model.train(X_train, y_train, epochs=100)
        y_hat, _ = model.forward(X_test)
        rmse = root_mean_squared_error(y_test, y_hat)
        r2 = r2_score(y_test, y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        evs = explained_variance_score(y_test, y_hat)
        mse = rmse ** 2
        return macro
    else:
        return('not supported model or u buggin')