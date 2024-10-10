import pandas as pd
df = pd.read_csv("hf://datasets/gauss314/options-IV-SP500/data_IV_USA.csv")
df.to_csv('data/options.csv', index = False)