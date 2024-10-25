import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
from deep_models import OptionsNN  # Assume your original file is named options_nn.py
import logging
from preprocess import preprocess
import pandas as pd

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Specify the format of log messages
    handlers=[
        logging.FileHandler("tuning.log"),  # Log to a file named tuning.log
        logging.StreamHandler()  # Also output logs to the console
    ]
)

# Define the objective function for optuna to minimize test loss
def objective(trial):
    # Suggest hyperparameters to tune
    input_size = 22  # Based on your fixed input features size
    eta = trial.suggest_loguniform('eta', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    init_type = trial.suggest_categorical('init_type', ['kaiming', 'xavier', None])
    epochs = trial.suggest_int('epochs', 10, 100)

    # Create the model with suggested hyperparameters
    model = OptionsNN(input_size=input_size, eta=eta, init_type=init_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv("../data/options.csv")  # load data
    df_subsample = df.sample(frac=0.3, random_state=69)  # ~400,000
    X_train_scaled, y_train, X_test_scaled, y_test = preprocess(df_subsample, remove_outliers=True)
    X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    train_dataset = TensorDataset(X_train_scaled, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    train_size = len(train_dataset)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(train_size).to(device)  # Shuffle training data
        for i in range(0, train_size, batch_size):
            idxs = perm[i:i + batch_size]
            x_bat, y_bat = X_train_scaled[idxs], y_train[idxs]
            x_bat, y_bat = x_bat.to(device), y_bat.to(device)

            model.optimizer.zero_grad()
            y_pred = model(x_bat)
            loss = model.criterion(y_pred, y_bat)
            loss.backward()
            model.optimizer.step()

    # Evaluation on the test set after training
    test_loss = model.evaluate(X_test_scaled, y_test)

    # Log the trial's result
    logging.info(f'Trial complete - Test loss: {test_loss:.4f}')

    # Return the test loss for optuna to minimize
    return test_loss


# Run the hyperparameter tuning process
def run_tuning():
    logging.info("Starting hyperparameter tuning to minimize test loss.")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Print and log the best hyperparameters
    logging.info(f"Best hyperparameters: {study.best_params}")
    logging.info(f"Best test loss: {study.best_value}")


if __name__ == "__main__":
    run_tuning()
