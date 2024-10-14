# import logging
# from models import model_picker
# import pandas as pd
# from preprocess import preprocess
#
# print('imports done')
#
# logging.basicConfig(level=print, format='%(asctime)s - %(levelname)s - %(message)s')
#
# df = pd.read_csv("data/options.csv")
# X_train_scaled, y_train, X_test_scaled, y_test = preprocess(df)
# print('df done')
#
# print("Running Support Vector Regression model...")
# svr_results = model_picker('Support Vector Regression',
#                            X_train=X_train_scaled,
#                            y_train=y_train,
#                            X_test=X_test_scaled,
#                            y_test=y_test)
# print(f"SVR Results: {svr_results}")