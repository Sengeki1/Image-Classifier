import numpy as np 

class SVM:

    def __init__(self,learning_rate = 0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, x_train, y_train):
        n_samples, n_features = x_train.shape # returns a tuple with the dimension and number of elements
        y = np.where(y_train <= 0, -1, 1) # returns y_train as x if condition is true

        # init weights
        self.w = np.zeros(n_features)
        self.b = 0

        # learn the weights with the update rule formula
        for _ in range(self.n_iters):
            print("Sample = ", _, end="\r")
            for index, x_i in enumerate(x_train): # for all the samples
                if ((y[index] * (np.dot(x_i, self.w) - self.b)) >= 1):
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[index]))
                    self.b -= self.lr * y[index]

    def predict(self, x_train):
        approx = np.dot(x_train, self.w) - self.b
        return np.sign(approx) # if array value is greater than 0 it returns 1, if 
        # array value is less than 0 it returns -1, and if array value 0 it returns 0.