import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

def sigmoid(z):
    return 1/(1+np.exp(-z))

class SampleData():
    def __init__(self,x,y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x = x
        self.y = y
    
class LogisticRegression(SampleData):
    def __init__(self,x,y,alpha=0.01,iterations=10000,lambda_=0.01):
        super().__init__(x,y)
        self.alpha=alpha
        self.iterations=iterations
        self.lambda_=lambda_
        self.m = self.x.shape[0]
        self.weights = np.zeros((self.x.shape[1], 1))
        self.bias = np.zeros((1, 1))

    def get_gradient(self):
        self.z = self.x@self.weights + self.bias
        self.y_pred = np.clip(sigmoid(self.z), 1e-9, 1 - 1e-9)
        error = self.y_pred - self.y
        dw = (1/self.m) * self.x.T @ error + (self.lambda_/self.m)*self.weights
        db = (1/self.m) * np.sum(error)
        return dw, db

    def gradient_descent(self):
        for i in range(self.iterations):
            dw, db = self.get_gradient()
            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db
            
            if i % 500 == 0:
                loss = (-1/self.m) * (self.y.T @ np.log(self.y_pred) + (1 - self.y).T @ np.log(1 - self.y_pred))+ (self.lambda_/2)*self.weights.T@self.weights
                print(f"Iteration {i}: Loss = {loss}")
            
            if i == 6000:
                self.alpha = 0.001

        return self.weights, self.bias, loss

bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features 
x = pd.get_dummies(X, dtype=float)
x = (x-x.mean(axis=0))/(x.std(axis=0)+1e-9)
y = bank_marketing.data.targets
y = y.replace({'no': 0, 'yes': 1}).astype(float)
gd = LogisticRegression(x, y, alpha=0.01, iterations=10000, lambda_=0.01)
print("*******Training started*******")
optimal_weights, optimal_bias, final_loss = gd.gradient_descent()
print("*******Training completed*******")
print(f"final loss: {final_loss}")
print(f"Optimal Weights: {optimal_weights.flatten()}")
print(f"Optimal Bias: {optimal_bias}")