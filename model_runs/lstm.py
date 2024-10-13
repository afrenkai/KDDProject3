import logging
from models import model_picker
import pandas as pd
from preprocess import preprocess
from harder_models import LSTM
print('imports done')

logging.basicConfig(level=print, format='%(asctime)s - %(levelname)s - %(message)s')

df = pd.read_csv("../data/options.csv")
X_train_scaled, y_train, X_test_scaled, y_test = preprocess(df)
print('df done')

# LSTM
print("Running Long Short Term Memory model...")
lstm_results = model_picker('Long Short Term Memory',
                            X_train=X_train_scaled,
                            y_train=y_train,
                            X_test=X_test_scaled,
                            y_test=y_test)
print(f"LSTM Results: {lstm_results}")