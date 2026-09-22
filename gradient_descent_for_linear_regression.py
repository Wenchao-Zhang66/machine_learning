import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split as tts


class SampleData:
    def __init__(self, x, y):
        self.loss = None
        self.gradient = None
        self.y = np.asarray(y, dtype=float)  # target
        bias = np.ones((x.shape[0], ))
        self.x = np.empty((x.shape[0], x.shape[1] + 1))
        self.x[:,1:] = x
        self.x[:,0] = bias
        self.m = self.x.shape[0]  # number of training examples
        self.n = self.x.shape[1]  # number of features
        self.theta = np.zeros((self.n, 1))  # weight parameters

    # compute the gradient
    def get_gradient(self):
        error = self.x @ self.theta - self.y
        self.gradient = 1.0 / self.m * (self.x.T @ error)
        return self.gradient

    # compute the loss function
    def loss_function(self):
        error = self.x @ self.theta - self.y
        self.loss = 1.0 / self.m * error.T @ error
        return self.loss


class GradientDescent(SampleData):
    def __init__(self, x, y, alpha=0.01, iterations=10000):
        super().__init__(x, y)
        self.alpha = alpha  # learning rate
        self.iterations = iterations

    def iterate(self):
        for i in range(self.iterations):
            self.theta -= self.alpha * self.get_gradient()
            if i % 500 == 0:
                print(f"Iteration {i}: Loss = {self.loss_function()}")
        return self.theta


def model_evaluation(x, y, theta):
    y = np.asarray(y, dtype=float)
    bias = np.ones((x.shape[0], ))
    x_test = np.empty((x.shape[0], x.shape[1] + 1))
    x_test[:, 1:] = x
    x_test[:, 0] = bias
    m_test = x.shape[0]  # number of test samples
    y_pred = x_test @ theta  # prediction
    y_avg = y.mean()  # average of test set
    error_test = y_pred - y
    mse = 1.0 / m_test * error_test.T @ error_test  # compute mean squared error
    ss_res = (y - y_pred).T @ (y - y_pred)  # residual sum of squares
    ss_tot = (y - y_avg).T @ (y - y_avg)  # total sum of squares
    r2 = 1.0 - ss_res / ss_tot  # r-squared
    return mse, r2


def get_data():
    real_estate_valuation = fetch_ucirepo(id=477)
    X = real_estate_valuation.data.features
    y = real_estate_valuation.data.targets
    X_tr, X_te, y_tr, y_te = tts(X, y,
                                 test_size=0.3, random_state=20260922)
    X_tr_avg = X_tr.mean(axis=0)
    X_tr_std = X_tr.std(axis=0)
    X_tr = (X_tr - X_tr_avg) / X_tr_std
    X_te = (X_te - X_tr_avg) / X_tr_std
    print(f"the number of training samples: {X_tr.shape[0]}")
    print(f"the number of features: {X.shape[1]}")
    return X_tr, X_te, y_tr, y_te


if __name__ == "__main__":
    # tr: train; te: test
    X_tr, X_te, y_tr, y_te = get_data()
    print("*******Training started*******")
    gd = GradientDescent(X_tr, y_tr, alpha=0.01, iterations=10000)
    optimal_theta = gd.iterate()
    print("*******Training completed*******")
    print("Optimal theta:", optimal_theta)
    print("Final Loss:", gd.loss_function())
    print("="*50)
    print("*******Testing started*******")
    print(f"the number of test samples: {X_te.shape[0]}")
    mse, r2 = model_evaluation(X_te, y_te, optimal_theta)
    print("******Testing completed*******")
    print(f"MSE: {mse}")
    print(f"R2: {r2}")
