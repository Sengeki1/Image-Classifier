import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt( np.sum((x1-x2) ** 2 ))
    return distance


class KNearestNeighbour:
    def __init__(self, k=3):
        self.k = k
        self.sample = 0
    
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x_train):
        predictions = [self._predict(x) for x in x_train] # predict each sample from the training 
        return predictions
    
    def _predict(self, x):
        self.sample += 1
        print("Sample = ", self.sample, end="\r") 
        # compute the distance between the given point and the other points
        distances = [euclidean_distance(x, x_train) for x_train in self.x_train]

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        # argsort sorts the array keeping the indices of the previous arry before sorted and then
        # we get the closest k neighboors that would be, getting the first indexes
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # our label to check which classe our sample is more likely to be

        most_common = Counter(k_nearest_labels).most_common() # getting the most common classe
        return most_common[0][0]