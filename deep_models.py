import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

class SGDRegressor:
    def __init__(self, eta=0.001, epochs=1000):
        self.lr = eta
        self.epochs = epochs
        self.w = None
        self.b = None
        logging.info(f'Initialized SGD Regression with eta of {self.lr}')

    def predict(self, X):
        logging.info('predicting With SGD')
        return np.dot(X, self.w) + self.b

    def fit(self, X, y):
        n, m = X.shape
        logging.info(f'Training began with {n} samples and {m} features')
        y = y.to_numpy()
        self.w = np.zeros(m)
        self.b = 0

        for epoch in range(self.epochs):
            for i in range(n):
                y_hat = np.dot(X[i], self.w) + self.b
                dw = (y_hat - y[i]) * X[i]
                db = (y_hat - y[i])

                self.w -= self.lr * dw
                self.b -= self.lr * db
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                logging.info(f"Epoch {epoch+1}/{self.epochs}: Weights: {self.w}, Bias: {self.b}")

    @staticmethod
    def mse(y_true, y_hat):
        return np.mean((y_true - y_hat)) ** 2


# class LSTM:
#     def __init__(self, input_size, hidden_size, output_size, eta=0.001):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.lr = eta
#         self.Wf = np.random.randn(hidden_size, hidden_size + input_size)
#         self.Wi = np.random.randn(hidden_size, hidden_size + input_size)
#         self.Wo = np.random.randn(hidden_size, hidden_size + input_size)
#         self.Wc = np.random.randn(hidden_size, hidden_size + input_size)
#         self.bf = np.zeros((hidden_size, 1))
#         self.bi = np.zeros((hidden_size, 1))
#         self.bo = np.zeros((hidden_size, 1))
#         self.bc = np.zeros((hidden_size, 1))
#         self.Wy = np.random.randn(output_size, hidden_size)
#         self.by = np.zeros((output_size, 1))
#
#         logging.info(f'initialized LSTM with input size of {input_size}, '
#                      f'hidden size of {hidden_size}, '
#                      f'output size of {output_size}, '
#                      f'and learning rate of {self.lr}')
#
#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))
#
#     def sigmoid_derivative(self, x):
#         return self.sigmoid(x) * (1 - self.sigmoid(x))
#
#     def tanh(self, x):
#         return np.tanh(x)
#
#     def tanh_prime(self, x):
#         return 1 - np.tanh(x) ** 2
#
#     def forward(self, X):
#         logging.info('starting fwd pass for LSTM')
#         h_t = np.zeros((self.hidden_size, 1))
#         C_t = np.zeros((self.hidden_size, 1))
#         self.cache = {'h': [], 'C': [], 'f': [], 'i': [], 'o': [], 'C_tilde': [], 'x': []}
#
#         outputs = []
#         for t in range(X.shape[0]):
#             x_t = X[t].reshape(-1, 1)
#             combined = np.vstack((h_t, x_t))
#             f_t = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
#             i_t = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
#             C_tilde_t = self.tanh(np.dot(self.Wc, combined) + self.bc)
#             C_t = f_t * C_t + i_t * C_tilde_t
#             o_t = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
#             h_t = o_t * self.tanh(C_t)
#             y_t = np.dot(self.Wy, h_t) + self.by
#             outputs.append(y_t)
#             self.cache['h'].append(h_t)
#             self.cache['C'].append(C_t)
#             self.cache['f'].append(f_t)
#             self.cache['i'].append(i_t)
#             self.cache['o'].append(o_t)
#             self.cache['C_tilde'].append(C_tilde_t)
#             self.cache['x'].append(combined)
#         logging.info('fwd pass done')
#         return np.array(outputs).squeeze()
#
#     def backward(self, X, y_true, y_pred):
#         logging.info('starting bck prop for LSTM')
#         dWf, dWi, dWo, dWc = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wo), np.zeros_like(self.Wc)
#         dbf, dbi, dbo, dbc = np.zeros_like(self.bf), np.zeros_like(self.bi), np.zeros_like(self.bo), np.zeros_like(self.bc)
#         dWy, dby = np.zeros_like(self.Wy), np.zeros_like(self.by)
#         dh_next = np.zeros((self.hidden_size, 1))
#         dC_next = np.zeros((self.hidden_size, 1))
#
#         for t in reversed(range(X.shape[0])):
#             combined = self.cache['x'][t]
#             h_t = self.cache['h'][t]
#             C_t = self.cache['C'][t]
#             f_t = self.cache['f'][t]
#             i_t = self.cache['i'][t]
#             o_t = self.cache['o'][t]
#             C_tilde_t = self.cache['C_tilde'][t]
#             dy = y_pred[t] - y_true[t]
#             dWy += np.dot(dy, h_t.T)
#             dby += dy
#             dh = np.dot(self.Wy.T, dy) + dh_next
#             do = dh * np.tanh(C_t)
#             do_raw = self.sigmoid_derivative(o_t) * do
#             dWo += np.dot(do_raw, combined.T)
#             dbo += do_raw
#             dC = dh * o_t * self.tanh_prime(C_t) + dC_next
#             dC_tilde = dC * i_t
#             dC_tilde_raw = self.tanh_prime(C_tilde_t) * dC_tilde
#             dWc += np.dot(dC_tilde_raw, combined.T)
#             dbc += dC_tilde_raw
#             di = dC * C_tilde_t
#             di_raw = self.sigmoid_derivative(i_t) * di
#             dWi += np.dot(di_raw, combined.T)
#             dbi += di_raw
#             df = dC * C_t
#             df_raw = self.sigmoid_derivative(f_t) * df
#             dWf += np.dot(df_raw, combined.T)
#             dbf += df_raw
#             dC_next = dC * f_t
#             dh_next = np.dot(self.Wf.T, df_raw) + np.dot(self.Wi.T, di_raw) + np.dot(self.Wc.T, dC_tilde_raw) + np.dot(self.Wo.T, do_raw)
#
#         self.Wf -= self.lr * dWf
#         self.Wi -= self.lr * dWi
#         self.Wo -= self.lr * dWo
#         self.Wc -= self.lr * dWc
#         self.bf -= self.lr * dbf
#         self.bi -= self.lr * dbi
#         self.bo -= self.lr * dbo
#         self.bc -= self.lr * dbc
#         self.Wy -= self.lr * dWy
#         self.by -= self.lr * dby
#
#         logging.info('backprop done')
#
#     def train(self, X, y_true, epochs=10):
#         logging.info(f'training for {epochs} epochs')
#         for epoch in range(epochs):
#             y_pred = self.forward(X)
#             loss = np.mean((y_pred - y_true) ** 2)
#             logging.info(f'epoch {epoch + 1} / epochs, Loss : {loss}')
#             self.backward(X, y_true, y_pred)
#         logging.info('training done for LSTM')

# class DNN:
#     def __init__(self, layer_sizes, eta=0.01):
#         self.layer_sizes = layer_sizes
#         self.lr = eta
#         self.params = self.initialize_params()
#         logging.info(f'initialized DNN with layer_sizes of {layer_sizes} and learning rate of {self.lr}')
#     def initialize_params(self):
#         np.random.seed(69)
#         params = {}
#         num_layers = len(self.layer_sizes)
#         logging.info('beginning kaiming intialization')
#
#
#         for l in range(1, num_layers):
#             params['W' + str(l)] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * np.sqrt(2 / self.layer_sizes[l-1])
#             params['b' + str(l)] = np.zeros((self.layer_sizes[l], 1))
#         return params
#
#     def sigmoid(self, Z):
#         return 1 / (1 + np.exp(-Z))
#
#     def sigmoid_derivative(self, Z):
#         return self.sigmoid(Z) * (1 - self.sigmoid(Z))
#
#     def relu(self, Z):
#         return np.maximum(0, Z)
#
#     def relu_derivative(self, Z):
#         return np.where(Z > 0, 1, 0)
#
#     def forward(self, X):
#         X = X.T  # Transpose input
#         logging.info('starting DNN forward pass')
#
#         caches = {}
#         A = X
#         L = len(self.layer_sizes) - 1
#
#         for l in range(1, L):
#             logging.info(f'Shape of W{l}: {self.params["W" + str(l)].shape}')
#             logging.info(f'Shape of A{l - 1}: {A.shape}')
#             Z = np.dot(self.params['W' + str(l)], A) + self.params['b' + str(l)]
#             A = self.relu(Z)
#             caches['A' + str(l)] = A
#             caches['Z' + str(l)] = Z
#
#         ZL = np.dot(self.params['W' + str(L)], A) + self.params['b' + str(L)]
#         AL = ZL  # No activation for regression
#
#         caches['A' + str(L)] = AL
#         caches['Z' + str(L)] = ZL
#         logging.info('fwd pass done')
#
#         return AL, caches
#
#     def backward(self, X, y, caches):
#         logging.info('starting backprop for DNN')
#
#         grads = {}
#         m = X.shape[1]
#         L = len(self.layer_sizes) - 1
#         AL = caches['A' + str(L)]
#
#         # Compute gradient of AL with respect to loss
#         dAL = (AL - y) / m
#         logging.info(f'Shape of dAL: {dAL.shape}')
#
#         grads['dW' + str(L)] = np.dot(dAL, caches['A' + str(L - 1)].T)
#         grads['db' + str(L)] = np.sum(dAL, axis=1, keepdims=True)
#
#         for l in reversed(range(1, L)):
#             dZ = np.dot(self.params['W' + str(l + 1)].T, dAL) * self.relu_derivative(caches['Z' + str(l)])
#             logging.info(f'Shape of dZ at layer {l}: {dZ.shape}')
#
#             dW = np.dot(dZ, caches['A' + str(l - 1)].T if l > 1 else X.T)
#             db = np.sum(dZ, axis=1, keepdims=True)
#
#             grads['dW' + str(l)] = dW
#             grads['db' + str(l)] = db
#             dAL = dZ
#         logging.info('backprop done for DNN')
#         return grads
#
#     def update_parameters(self, grads):
#         logging.info('updating params')
#         L = len(self.layer_sizes) - 1
#         for l in range(1, L+1):
#             self.params['W' + str(l)] -= self.lr * grads['dW' + str(l)]
#             self.params['b' + str(l)] -= self.lr * grads['db' + str(l)]
#         logging.info('update complete')
#
#     def compute_loss(self, y_pred, y_true):
#         logging.info(f'Prediction shape: {y_pred.shape}, True label shape: {y_true.shape}')
#
#         # Reshape y_true to match y_pred if necessary
#         if y_true.shape != y_pred.shape:
#             y_true = y_true.reshape(y_pred.shape)
#
#         return np.mean((y_pred - y_true) ** 2)
#
#     def train(self, X, y, epochs=100):
#         logging.info(f'Training DNN for {epochs} epochs')
#
#         # Transpose y if it doesn't match the shape of y_pred
#         if y.shape[0] != self.layer_sizes[-1]:
#             y = y.T  # Transpose to match y_pred
#
#         for epoch in range(epochs):
#             y_pred, caches = self.forward(X)
#             loss = self.compute_loss(y_pred, y)
#             print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')
#             grads = self.backward(X, y, caches)
#             self.update_parameters(grads)
#         logging.info('training done for DNN')

class OptionsNN(nn.Module):
    def __init__(self, input_size = 13, eta=0.01, init_type = None):
        super(OptionsNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128) #fully connected layer 1. 13 feats currently in preprocessing, so 13 is input_size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        # shoutout my goat kaiming bro he cooked
        if init_type == 'kaiming':
            self._initialize_weights_kaiming()
        elif init_type == 'xavier':
            self._initialize_weights_xavier()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = eta)
        logging.info("OptionsNN model initialized with input size: %d, learning rate: %f", input_size, eta)

        self.to(device)
    def _initialize_weights_kaiming(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity = 'relu')
        logging.info("Goated Kaiming Methods Done")
    def _initialize_weights_xavier(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        logging.info('Cringe Xavier Method')
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_features)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def train_model(self, X_train, y_train, epochs=50, batch_size=32, val_data=None):
        train_size = X_train.size(0)
        logging.info("Training started for %d epochs with batch size %d", epochs, batch_size)


        X_train = X_train.to(device)
        y_train = y_train.to(device)


        progress_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")

        for epoch in progress_bar:
            self.train()
            perm = torch.randperm(train_size).to(device)  # Move permutation to GPU
            epoch_loss = 0

            for i in range(0, train_size, batch_size):
                idxs = perm[i:i + batch_size]
                x_bat, y_bat = X_train[idxs], y_train[idxs]

                self.optimizer.zero_grad()
                y_pred = self.forward(x_bat)
                loss = self.criterion(y_pred, y_bat)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / (train_size // batch_size)
            progress_bar.set_postfix(loss=avg_loss)

            if (epoch + 1) % 10 == 0:
                logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')


        logging.info("Training completed.")
    def evaluate(self, X_test, y_test):
        self.eval()
        logging.info("Evaluation started on test data.")

        # Move test data to GPU
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        with torch.no_grad():
            y_test_pred = self.forward(X_test)
            test_loss = self.criterion(y_test_pred, y_test)
            logging.info(f'Test Loss (MSE): {test_loss.item():.4f}')
        return test_loss.item()

    def predict(self, x):
        self.eval()

        # Move input to GPU
        x = x.to(device)

        with torch.no_grad():
            y_pred = self.forward(x)

        # Move output back to CPU for compatibility with NumPy
        return y_pred.cpu()

class OptionsLSTM:
    def __init__(self, input_size, hidden_size = 64, num_layers = 2, eta = 0.001):
        super(OptionsLSTM, self).__init__()
        self.model = tf.keras.models.Sequential()
        for i in range (num_layers):
            return_sequences = i < (num_layers - 1)  # Only return sequences for layers before the last
            self.model.add(layers.LSTM(hidden_size, return_sequences=return_sequences, input_shape=(None, input_size)))
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=eta), loss='mean_squared_error')
        self.model.summary()


        def train_model(self, X_train, y_train, epochs=50, batch_size=32, val_data=None):
            logging.info(f"Training model for {epochs} epochs with batch size {batch_size}")

            history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=val_data,
                                     verbose=1)

            return history


        def evaluate(self, X_test, y_test):
            logging.info("Evaluating model on test data")
            test_loss = self.model.evaluate(X_test, y_test, verbose=1)
            logging.info(f'Test Loss (MSE): {test_loss:.4f}')
            return test_loss


        def predict(self, X):
            logging.info("Making predictions")
            return self.model.predict(X)