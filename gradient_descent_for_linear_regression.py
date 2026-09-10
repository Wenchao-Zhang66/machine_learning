import numpy as np
from ucimlrepo import fetch_ucirepo

class SampleData:
    def __init__(self,x,y):
        x = np.asarray(x, dtype=float)#avoid x being a list
        y = np.asarray(y, dtype=float)#avoid y being a list
        x_0=np.ones((x.shape[0], 1))
        self.x = np.hstack((x_0, x))#set bias term = 1
        self.y = y
        self.m=self.x.shape[0]#the number of training examples
        self.n=self.x.shape[1]#the number of features
        self.theta=np.zeros((self.n,1))#initialize parameters
    #compute the gradient
    def get_gradient(self):
        error = self.x @ self.theta - self.y
        self.gradient = (1/self.m) * (self.x.T @ error)
        return self.gradient
    #compute the loss function
    def loss_function(self):
        self.loss=(1/(2*self.m))*(self.x@self.theta-self.y).T@(self.x@self.theta-self.y)
        return self.loss
#iteratively update the parameters using gradient descent
class GradientDescent(SampleData):
    def __init__(self,x,y,alpha=0.01,iterations=10000):
        super().__init__(x,y)
        self.alpha=alpha
        self.iterations=iterations
    
    def iterate(self):
        for i in range(self.iterations):
            self.theta=self.theta-self.alpha*self.get_gradient()
              
            if i % 500 == 0:
                print(f"Iteration {i}: Loss = {self.loss_function()}")
        
        return self.theta

real_estate_valuation = fetch_ucirepo(id=477) 
X_original = real_estate_valuation.data.features 
X=(X_original-X_original.mean(axis=0))/X_original.std(axis=0)
y = real_estate_valuation.data.targets
gd = GradientDescent(X, y, alpha=0.01, iterations=10000)
Optimal_theta = gd.iterate()
print("*******Training completed*******")
print("Optimal theta:", Optimal_theta)
print("Final Loss:", gd.loss_function())