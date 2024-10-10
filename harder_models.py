import numpy as np
import logging

class SGDRegressor():
    def __init__ (self, eta = 0.001, epochs = 1000):
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
            for i in range (n):
                y_hat = np.dot(X[i], self.w) + self.b
                dw = (y_hat - y[i]) * X[i]
                db = (y_hat -  y[i])

                self.w -= self.lr * dw
                self.b -= self.lr * db
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                logging.info(f"Epoch {epoch+1}/{self.epochs}: Weights: {self.w}, Bias: {self.b}")

    def mse(self, y_true, y_hat):
        return np.mean((y_true - y_hat)) ** 2

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, eta=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = eta
        self.Wf = np.random.randn(hidden_size, hidden_size + input_size)
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size)
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size)
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size)
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.Wy = np.random.randn(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, X):
        h_t = np.zeros((self.hidden_size, 1))
        C_t = np.zeros((self.hidden_size, 1))
        self.cache = {'h': [], 'C': [], 'f': [], 'i': [], 'o': [], 'C_tilde': [], 'x': []}
        
        outputs = []
        for t in range(X.shape[0]):
            x_t = X[t].reshape(-1, 1)
            combined = np.vstack((h_t, x_t))
            f_t = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
            i_t = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
            C_tilde_t = self.tanh(np.dot(self.Wc, combined) + self.bc)
            C_t = f_t * C_t + i_t * C_tilde_t
            o_t = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
            h_t = o_t * self.tanh(C_t)
            y_t = np.dot(self.Wy, h_t) + self.by
            outputs.append(y_t)
            self.cache['h'].append(h_t)
            self.cache['C'].append(C_t)
            self.cache['f'].append(f_t)
            self.cache['i'].append(i_t)
            self.cache['o'].append(o_t)
            self.cache['C_tilde'].append(C_tilde_t)
            self.cache['x'].append(combined)
        
        return np.array(outputs).squeeze()

    def backward(self, X, y_true, y_pred):
        dWf, dWi, dWo, dWc = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wo), np.zeros_like(self.Wc)
        dbf, dbi, dbo, dbc = np.zeros_like(self.bf), np.zeros_like(self.bi), np.zeros_like(self.bo), np.zeros_like(self.bc)
        dWy, dby = np.zeros_like(self.Wy), np.zeros_like(self.by)
        dh_next = np.zeros((self.hidden_size, 1))
        dC_next = np.zeros((self.hidden_size, 1))
        
        for t in reversed(range(X.shape[0])):
            combined = self.cache['x'][t]
            h_t = self.cache['h'][t]
            C_t = self.cache['C'][t]
            f_t = self.cache['f'][t]
            i_t = self.cache['i'][t]
            o_t = self.cache['o'][t]
            C_tilde_t = self.cache['C_tilde'][t]
            dy = y_pred[t] - y_true[t]
            dWy += np.dot(dy, h_t.T)
            dby += dy
            dh = np.dot(self.Wy.T, dy) + dh_next
            do = dh * np.tanh(C_t)
            do_raw = self.sigmoid_derivative(o_t) * do
            dWo += np.dot(do_raw, combined.T)
            dbo += do_raw
            dC = dh * o_t * self.tanh_prime(C_t) + dC_next
            dC_tilde = dC * i_t
            dC_tilde_raw = self.tanh_prime(C_tilde_t) * dC_tilde
            dWc += np.dot(dC_tilde_raw, combined.T)
            dbc += dC_tilde_raw
            di = dC * C_tilde_t
            di_raw = self.sigmoid_derivative(i_t) * di
            dWi += np.dot(di_raw, combined.T)
            dbi += di_raw
            df = dC * C_t
            df_raw = self.sigmoid_derivative(f_t) * df
            dWf += np.dot(df_raw, combined.T)
            dbf += df_raw
            dC_next = dC * f_t
            dh_next = np.dot(self.Wf.T, df_raw) + np.dot(self.Wi.T, di_raw) + np.dot(self.Wc.T, dC_tilde_raw) + np.dot(self.Wo.T, do_raw)

        self.Wf -= self.lr * dWf
        self.Wi -= self.lr * dWi
        self.Wo -= self.lr * dWo
        self.Wc -= self.lr * dWc
        self.bf -= self.lr * dbf
        self.bi -= self.lr * dbi
        self.bo -= self.lr * dbo
        self.bc -= self.lr * dbc
        self.Wy -= self.lr * dWy
        self.by -= self.lr * dby

    def train(self, X, y_true, epochs=10):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = np.mean((y_pred - y_true) ** 2)
            print(f'Epoch {epoch+1}, Loss: {loss}')
            self.backward(X, y_true, y_pred)


class DNN:
    def __init__(self, layer_sizes, eta=0.01):
        self.layer_sizes = layer_sizes
        self.lr = eta
        self.params = self.initialize_params()

    def initialize_params(self):
        np.random.seed(69)
        params = {}
        num_layers = len(self.layer_sizes)
        
        # shoutout my goat kaiming bro he cooked
        for l in range(1, num_layers):
            params['W' + str(l)] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * np.sqrt(2 / self.layer_sizes[l-1])
            params['b' + str(l)] = np.zeros((self.layer_sizes[l], 1))
        
        return params

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)

    def forward(self, X):
        caches = {}
        A = X
        L = len(self.layer_sizes) - 1

        for l in range(1, L):
            Z = np.dot(self.parameters['W' + str(l)], A) + self.parameters['b' + str(l)]
            A = self.relu(Z)
            caches['A' + str(l)] = A
            caches['Z' + str(l)] = Z
        
        ZL = np.dot(self.parameters['W' + str(L)], A) + self.parameters['b' + str(L)]
        AL = ZL
        caches['A' + str(L)] = AL
        caches['Z' + str(L)] = ZL

        return AL, caches

    def backward(self, X, y, caches):
        grads = {}
        m = X.shape[1]
        L = len(self.layer_sizes) - 1
        AL = caches['A' + str(L)]
        dAL = (AL - y) / m
        grads['dW' + str(L)] = np.dot(dAL, caches['A' + str(L-1)].T)
        grads['db' + str(L)] = np.sum(dAL, axis=1, keepdims=True)
        for l in reversed(range(1, L)):
            dZ = np.dot(self.parameters['W' + str(l+1)].T, dAL) * self.relu_derivative(caches['Z' + str(l)])
            dW = np.dot(dZ, caches['A' + str(l-1)].T if l > 1 else X.T)
            db = np.sum(dZ, axis=1, keepdims=True)
            
            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db
            dAL = dZ
        return grads

    def update_parameters(self, grads):
        L = len(self.layer_sizes) - 1
        for l in range(1, L+1):
            self.parameters['W' + str(l)] -= self.lr * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= self.lr * grads['db' + str(l)]

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            y_pred, caches = self.forward(X)
            
            loss = self.compute_loss(y_pred, y)
            print(f'Epoch {epoch+1}, Loss: {loss}')
            
            grads = self.backward(X, y, caches)
            self.update_parameters(grads)

