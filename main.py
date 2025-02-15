import numpy as np
import matplotlib.pyplot as plt
from data import Data
from knn import KNearestNeighbour
from svm import SVM
from softmax import Softmax
from twoLayerNeuralNet import NeuralNet

if __name__ == "__main__":
    main = Data()
    data = main.load_Data('cifar-10-batches-py')
    
    # visualizing 
    sample = data['test'][98] # taking a sample from the testing data

    R = sample[0:1024].reshape(32, 32)
    G = np.reshape(sample[1024:2048], newshape=(32, 32))
    B = np.reshape(sample[2048:], newshape=(32, 32))
    sample = np.dstack((R, G, B)) # stacking them together as a sequence depth wise (along third axis) 
    plt.imshow(sample)
    plt.show()

    x_test, y_test = data['test'], data['testlabel']
    # Splitting the data into train and validation set
    train_x, val_x, train_y, val_y = main.split_Data()
    
    # Classifiers

    #
    #   K-Nearest Neighbour
    #

    knn = KNearestNeighbour()
    knn.fit(train_x, train_y)
    predictions_knn = knn.predict(val_x)

    acc_knn = np.sum(predictions_knn == val_y) / len(val_y)
    print(f"\nAccuracy: {acc_knn}") # Accuracy: 0.276 as k = 3
    
    #
    #   Support Vector Machine (SVM) 
    #

    svm = SVM()
    svm.fit(train_x, train_y)
    predictions_svm = svm.predict(val_x)

    acc_svm = np.sum(predictions_svm == val_y) / len(val_y)
    print(f"\nAccuracy: {acc_svm}") # Accuracy: 0.119 

    #
    #   Softmax
    #

    softmax = Softmax(train_x, val_x, val_y, train_y)
    print(f"\nAccuracy: {softmax.accuracy():.2f}") # Accuracy: 0.09

    #
    #   Two Layer Neural Network
    #

    train_x, train_y = train_x.T, train_y.reshape(1, train_y.shape[0])
    neuralNet = NeuralNet(train_x, train_y)
    params, cost = neuralNet.fit()

    print(f"\nAccuracy: {neuralNet.accuracy():.2f}") 
    plt.plot(cost)
    plt.show()