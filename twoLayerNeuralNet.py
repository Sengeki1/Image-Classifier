import numpy as np
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

class NeuralNet:

    def __init__(self, x_train, y_train, hidden_size = 5):
        self.x_train = x_train
        self.y_train = y_train
        self.hidden_size = hidden_size

        self.params = self.setParam()

    def setParam(self):
        np.random.seed(3)
        input_size = self.x_train.shape[0]
        output_size = self.y_train.shape[0]

        # setting the weights and biases.
        w1 = np.random.rand(self.hidden_size, input_size) * np.sqrt ( 1 / input_size)
        b1 = np.zeros((self.hidden_size, 1))
        w2 = np.random.rand(output_size, self.hidden_size) * np.sqrt ( 1 / self.hidden_size)
        b2 = np.zeros((output_size, 1))

        return {'W1': w1, 'W2': w2, 'b1': b1, 'b2': b2}
    
    def forwardPropagation(self):
        Z1 = np.dot(self.params['W1'], self.x_train) + self.params['b1']
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.params['W2'], A1) + self.params['b2']
        y = sigmoid(Z2)

        return y, {'Z1': Z1, 'Z2': Z2, 'A1': A1, 'y': y}
    
    def cost(self, predict):
        m = self.y_train.shape[1]
        cost = -np.sum(np.multiply(np.log(predict), self.y_train) + np.multiply((1 - self.y_train), np.log(1 - predict))) / m
        return np.squeeze(cost)
    
    def backPropagation(self, cache):
        m = self.x_train.shape[1]
        dy = cache['y'] - self.y_train

        dW2 = (1 / m) * np.dot(dy, np.transpose(cache['A1']))
        db2 = (1 / m) * np.sum(dy, axis=1, keepdims=True)
        dZ1 = np.dot(np.transpose(self.params['W2']), dy) * (1 - np.power(cache['A1'], 2))
        dW1 = (1 / m) * np.dot(dZ1, np.transpose(self.x_train))
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def updateParam(self, gradients, learning_rate = 1.2):
        W1 = self.params['W1'] - learning_rate * gradients['dW1']
        b1 = self.params['b1'] - learning_rate * gradients['db1']
        W2 = self.params['W2'] - learning_rate * gradients['dW2']
        b2 = self.params['b2'] - learning_rate * gradients['db2']

        return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
    
    def fit(self, learning_rate = 0.3, number_of_iterations = 1000):
        cost = []
        for i in range(number_of_iterations):
            print("Sample = ", i, end="\r")
            y, cache = self.forwardPropagation()
            costit = self.cost(y)
            gradients = self.backPropagation(cache)
            params = self.updateParam(gradients, learning_rate)

            cost.append(costit)
        return params, cost, y
    
    def accuracy(self):
        y_pred, _ = self.forwardPropagation()
        acc = accuracy_score(self.y_train, y_pred)
        return acc