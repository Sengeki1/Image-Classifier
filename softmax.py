import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class Softmax:
    
    def __init__(self, x_train, x_test, y_test, y_train, learning_rate = 0.01, num_epocs = 1000):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.lr = learning_rate
        self.num_epochs = num_epocs
        
        self.standardize(self.x_train, self.x_test)

        # Number of classes
        self.numClasses = len(np.unique(self.y_train))

        # Add bias term to feature matrix
        self.x_train_bias = np.hstack((self.x_train, np.ones((self.x_train.shape[0], 1))))
        self.x_test_bias = np.hstack((self.x_test, np.ones((self.x_test.shape[0], 1))))

        # Initialize weights randomly
        self.num_features = self.x_test_bias.shape[1]
        self.weights = np.random.randn(self.num_features, self.numClasses)

        self.training()
    
    def standardize(self, x_train, x_test):
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(x_train)
        self.x_test = scaler.transform(x_test) 

    def training(self):
        for epoch in range(self.num_epochs):
            print("Sample = ", epoch, end="\r") 
            # Compute logits (linear combinations)
            logits = self.x_train_bias.dot(self.weights)

            # Apply softmax function on logits
            exp_logits = np.exp(logits)
            softmax_prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Compute gradient of cross-entropy loss with respect to weights
            gradients = self.x_train_bias.T.dot(softmax_prob - np.eye(self.numClasses)[self.y_train])

            self.weights -= self.lr * gradients

    def predict(self):
        test_logits = self.x_test_bias.dot(self.weights)
        test_softmax_probs = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=1, keepdims=True)
        
        y_pred = np.argmax(test_softmax_probs, axis=1)
        return y_pred
    
    def accuracy(self):
        acc = accuracy_score(self.y_test, self.predict())
        return acc