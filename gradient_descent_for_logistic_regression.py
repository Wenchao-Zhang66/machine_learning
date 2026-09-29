import numpy as np
import pandas as pd
from scipy.special import expit
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split as tts


class LogisticRegression:
    def __init__(self, x, y, lr=0.01, iterations=10000, lambda_=0.01):
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.y_pred = None
        self.z = None
        self.lr = lr  # learning rate
        self.iterations = iterations  # the number of iterations
        self.lambda_ = lambda_  # the regularization strength
        self.m = self.x.shape[0]  # the number of samples
        self.weights = np.zeros((self.x.shape[1], 1))  # initialize parameters
        self.bias = np.zeros((1, 1))

    def get_gradient(self):
        self.z = self.x @ self.weights + self.bias
        self.y_pred = np.clip(expit(self.z), 1e-9, 1 - 1e-9)  # prediction
        error = self.y_pred - self.y
        dw_reg = self.lambda_ * self.weights  # regularization term for weight
        dw = (1.0 / self.m) * self.x.T @ error + dw_reg  # gradient of the weight
        db = (1.0 / self.m) * np.sum(error)  # gradient of the bias
        return dw, db

    def gradient_descent(self):
        for i in range(self.iterations):
            dw, db = self.get_gradient()
            self.weights -= self.lr * dw
            self.bias -= self.lr * db  # gradient descent

            if i % 500 == 0:
                reg = (self.lambda_ / 2.0) * self.weights.T @ self.weights  # regularization term
                term1 = self.y.T @ np.log(self.y_pred)
                term2 = (1.0 - self.y).T @ np.log(1.0 - self.y_pred)
                loss = (-1.0 / self.m) * (term1 + term2) + reg
                print(f"Iteration {i}: Loss = {loss}")

            if i == 6000:
                self.lr = 0.001  # reduce the learning rate to reduce the step size

        return self.weights, self.bias, loss


def model_evaluation(x,y,weights,bias):
    x = np.array(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m_test = x.shape[0]  # number of test samples
    z = x @ weights + bias  # prediction
    y_avg = y.mean()  # average of test set
    y_pred = expit(z)  # probability of the positive class
    term1 = y.T @ np.log(y_pred)
    term2 = (1.0 - y).T @ np.log(1.0 - y_pred)
    log_loss = - 1.0 / m_test * (term1 + term2)
    return log_loss


def get_data():
    bank_marketing = fetch_ucirepo(id=222)  # import data
    x = bank_marketing.data.features
    x = pd.get_dummies(x, dtype=float)  # one-hot encoding
    y = bank_marketing.data.targets
    y = y.replace({'no': 0, 'yes': 1}).astype(float)  # convert strings to numbers
    x_tr, x_te, y_tr, y_te = tts(x, y,
                                 test_size=0.3, random_state=20260928)
    x_tr_avg = x_tr.mean(axis=0)  # average of X_train
    x_tr_std = x_tr.std(axis=0)  # standard deviation of X_train
    x_tr = (x_tr - x_tr_avg) / x_tr_std
    x_te = (x_te - x_tr_avg) / x_tr_std
    print(f"number of training samples: {x_tr.shape[0]}")
    print(f"number of features: {x.shape[1]}")
    return x_tr, x_te, y_tr, y_te


if __name__ == "__main__":
    x_tr, x_te, y_tr, y_te = get_data()  # tr: train; te: test
    gd = LogisticRegression(x_tr, y_tr, lr=0.01, iterations=10000, lambda_=0.01)
    print("*******Training started*******")
    optimal_weights, optimal_bias, final_loss = gd.gradient_descent()
    print("*******Training completed*******")
    print(f"final loss: {final_loss}")
    print(f"Optimal Weights: {optimal_weights}")
    print(f"Optimal Bias: {optimal_bias}")
    print("="*50)
    print("*******Testing started*******")
    print(f"number of testing samples: {x_te.shape[0]}")
    log_loss = model_evaluation(x_te, y_te, optimal_weights, optimal_bias)
    print("*******Testing completed*******")
    print(f"Log Loss: {log_loss}")
