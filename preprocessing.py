import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("hf://datasets/gauss314/options-IV-SP500/data_IV_USA.csv")

feat = ['strikes_spread', 'calls_contracts_traded', 'puts_contracts_traded', 
            'calls_open_interest', 'puts_open_interest', 'expirations_number', 
            'contracts_number', 'hv_20', 'hv_40', 'hv_60', 'hv_120', 'hv_180', 'VIX']

X = df[feat]
y = df['DITM_IV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
