import sys
# hacky import for hacky project structure
sys.path.append("../")
import pandas as pd
from preprocess import preprocess
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from functools import reduce
from math import ceil
from joblib import dump, cpu_count
from os import path, makedirs
from time import perf_counter
import model_params as MP
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet


ITERATION_MULTIPLIER = 0.3 # controls number of fits for the random search cv
SEP = "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"


def get_save_name(estimator_name:str, grid_search:bool|None, outliers_removed:bool):
    # save_name e.g. "random_forest-grid-outlier.sav"
    save_name = estimator_name.replace(" ", "_") 
    if grid_search == None:
        pass
    elif grid_search == True:
        save_name += "-grid"
    else:
        save_name += "-random"

    if outliers_removed:
        save_name += "-outlier"

    save_name += ".sav"
    return save_name

def save_model(regressor, estimator_name:str, grid_search:bool|None, outliers_removed:bool):
    model_dir = 'saved_models'
    if not path.exists(model_dir):
        makedirs(model_dir)
    save_name = get_save_name(estimator_name, grid_search, outliers_removed)
    save_name = model_dir + "/" + save_name
    print(f"Saving model to {save_name}")
    dump(regressor, save_name, compress=6)


def evaluate(estimator, X_train_scaled, y_train, X_test_scaled, y_test):
    y_hat_test = estimator.predict(X_test_scaled)
    y_hat_train = estimator.predict(X_train_scaled)
    rmse = root_mean_squared_error(y_test, y_hat_test)
    rmse_train = root_mean_squared_error(y_train, y_hat_train)
    r2 = r2_score(y_test, y_hat_test)
    r2_train = r2_score(y_train, y_hat_train)
    mae = mean_absolute_error(y_test, y_hat_test)
    evs = explained_variance_score(y_test, y_hat_test)
    mse = rmse ** 2
    mse_train = rmse_train ** 2
    print(f"RMSE: {rmse}, RMSE_train: {rmse_train}, MSE: {mse}, MSE_train: {mse_train}, \
          R2: {r2}, R2_train: {r2_train}, MAE: {mae}, EVS: {evs}")

# here X_full_scaled and Y_full_scaled means the full train sample rather than the subsample used
def tune_model(estimator, params: dict, estimator_name:str, X_train_scaled, X_test_scaled, y_train, y_test,
               outliers_removed:bool, X_full_scaled, y_full_scaled):
    total_fits_needed = reduce(lambda x, y: x*y,[len(param_values) for param_values in params.values()], 1)
    random_search_iter = ceil(total_fits_needed*ITERATION_MULTIPLIER)

    regressor = estimator()

    # use validation set for grid search
    _, X_val_scaled, _, y_val = train_test_split(X_train_scaled, y_train, test_size=0.3,
                                                                        random_state=69)

    print(f"Tuning hyperparams for {estimator_name}")

    print("Starting Grid Search with cv=3")
    grid_search = GridSearchCV(estimator=regressor, param_grid=params, 
                            scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1, refit=False)
    
    start_time = perf_counter()
    grid_search.fit(X_val_scaled, y_val)
    print(f"Grid Search with outliers_removed={outliers_removed} took {perf_counter()-start_time} secs.")
    best_grid_cv_params = grid_search.best_params_
    regressor = estimator()


    regressor.set_params(**best_grid_cv_params)
    regressor.fit(X_full_scaled,y_full_scaled)
    
    print("Best Params from grid search")
    print(best_grid_cv_params)
    print("Evaluating best model from grid search:")
    evaluate(regressor,X_train_scaled,y_train,X_test_scaled,y_test)
    save_model(regressor, estimator_name, True, outliers_removed)

    regressor = estimator()
    print("Starting Randomized Search with cv=3")
    random_search = RandomizedSearchCV(estimator=regressor, param_distributions=params, 
                                   n_iter=random_search_iter, scoring='neg_mean_squared_error', cv=3, 
                                   verbose=1, random_state=69, n_jobs=-1, refit=False)
    
    start_time = perf_counter()
    random_search.fit(X_train_scaled, y_train)
    print(f"Random Search with remove_outliers={outliers_removed} took {perf_counter()-start_time} secs.")

    best_random_cv_params = random_search.best_params_
    regressor = estimator()
    regressor.set_params(**best_random_cv_params)
    regressor.fit(X_full_scaled,y_full_scaled)
    print("Best Params from random search")
    print(best_random_cv_params)
    print("Evaluating best model from random search:")
    evaluate(regressor,X_train_scaled,y_train,X_test_scaled,y_test)
    save_model(regressor, estimator_name, False, outliers_removed)
    print(SEP) # easier to go through the logs


# used exclusively for OLS
def fit_model(regressor, estimator_name:str, X_train_scaled, X_test_scaled, y_train, y_test,
               outliers_removed:bool):
    start_time = perf_counter()
    regressor.fit(X_train_scaled, y_train)
    print(f"OLS fit with remove_outliers={outliers_removed} took {perf_counter()-start_time} secs.")

    evaluate(regressor,X_train_scaled,y_train,X_test_scaled,y_test)
    save_model(regressor, estimator_name, None, outliers_removed)
    print(SEP) # easier to go through the logs




if __name__ == "__main__":
    print(f"Running Tuning Script with {cpu_count()} CPU cores")
    # load data
    df = pd.read_csv("../data/options.csv") # load data
    df_subsample = df.sample(frac=0.25, random_state=69) # ~400,000
    hist_grad_boost_hack = True # sry this is bad code, but efficient in this case
    for remove_outliers in [False, True]:
        # preprocess data
        X_train_scaled_full, y_train_full, X_test_scaled_full, y_test_full = preprocess(df, remove_outliers=remove_outliers)

  
        # use subsample for the intensive training
        X_train_scaled, y_train, X_test_scaled, y_test = preprocess(df_subsample, remove_outliers=remove_outliers)

        if hist_grad_boost_hack:
            hist_grad_boost_hack = False
            # this is a hacky/lazy way of running an un-tuned model in the tune loop.
            fit_model(HistGradientBoostingRegressor(random_state=69), 
                       MP.HGB_NAME+" Initial", X_train_scaled_full, X_test_scaled_full, y_train_full, y_test_full,remove_outliers)

      # OLS
        fit_model(LinearRegression(n_jobs=-1), 
                       MP.OLS_NAME, X_train_scaled_full, X_test_scaled_full, y_train_full, y_test_full,remove_outliers)


        # elastic net
        tune_model(ElasticNet, MP.EN_PARAMS, MP.EN_NAME, X_train_scaled, 
                X_test_scaled, y_train, y_test, remove_outliers, X_train_scaled_full, y_train_full)
        
        # random forest
        tune_model(RandomForestRegressor, MP.RF_PARAMS, MP.RF_NAME, X_train_scaled, 
                X_test_scaled, y_train, y_test,remove_outliers, X_train_scaled_full, y_train_full)
        # histogram gradient boosting
        tune_model(HistGradientBoostingRegressor, MP.HGB_PARAMS, MP.HGB_NAME, X_train_scaled, 
                X_test_scaled, y_train, y_test, remove_outliers, X_train_scaled_full, y_train_full)
