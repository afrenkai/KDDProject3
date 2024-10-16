import logging
import pandas as pd
from preprocess import preprocess
from models import model_picker
import tensorflow as tf


print('imports done')

logging.basicConfig(level=print, format='%(asctime)s - %(levelname)s - %(message)s')

df = pd.read_csv("data/options.csv")
X_train, y_train, X_test, y_test = preprocess(df, seq=True, seq_len=10)
print('df done')

print("Running Long Short Term Memory model...")
lstm_results= model_picker(
    type='Long Short Term Memory',
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    seq_len=10
)
print(f"LSTM Results: {lstm_results}")
