import pandas as pd
import os

dir = 'data'
if not os.path.exists(dir):
    os.makedirs(dir)

df = pd.read_csv("hf://datasets/gauss314/options-IV-SP500/data_IV_USA.csv")
df.to_csv('data/options.csv', index = False)