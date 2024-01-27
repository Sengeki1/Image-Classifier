import numpy as np
import matplotlib.pyplot as plt
from data import Data
from knn import KNearestNeighbour

if __name__ == "__main__":
    main = Data()
    data = main.load_Data('cifar-10-batches-py')
    
    # visualizing 
    sample = data['test'][47] # taking a sample from the testing data

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
    knn = KNearestNeighbour(k = 5)
    knn.fit(train_x, train_y)
    predictions = knn.predict(val_x)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)
