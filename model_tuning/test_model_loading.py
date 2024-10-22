from tune_model import get_save_name, evaluate
from joblib import load
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import model_params as MP

def load_model(estimator_name:str, grid_search:bool, outliers_removed:bool):
    save_name = 'saved_models' + "/" + get_save_name(estimator_name, grid_search, outliers_removed)
    regressor = load(save_name)
    return regressor

if __name__ == "__main__":
    reg = load_model(MP.SVR_NAME, True, False)
    x, y = make_regression(n_samples=300, n_features=2, noise=10, random_state=69)
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                                      random_state=69)
    # use validation set for grid search
    _, X_val_scaled, _, y_val = train_test_split(X_train_scaled, y_train, test_size=0.3,
                                                                        random_state=69)
    evaluate(reg,X_train_scaled,y_train,X_test_scaled,y_test)
