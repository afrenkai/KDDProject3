import pandas as pd
df = pd.read_csv('data/options.csv')

df_new = df.sample(n = 50000)
df_new.to_csv('data/sample.csv', index = False)