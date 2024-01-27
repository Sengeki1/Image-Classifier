import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

class Data:

    def __init__(self):
        self.train = []
        self.trainlabel = []
        self.test = []
        self.testlabel = []

        self.dictData = {}

    def load_Data(self, path):
        files = os.listdir(path=path)

        for file in files[:6]:
            with open(path + '/' + file, 'rb') as fo: # since we are opening the files as binary
                dict = pickle.load(fo, encoding='bytes')
                if (file != 'test_batch'):
                    self.train.append(dict[b'data']) # the data should be saved as bytes
                    self.trainlabel.append(dict[b'labels'])
                else:
                    self.test.append(dict[b'data'])
                    self.testlabel.append(dict[b'labels'])

        self.dictData = {'train': np.array(self.train), 'trainlabel': np.array(self.trainlabel), 
                    'test': np.array(self.test), 'testlabel': np.array(self.testlabel)}
        
        # We reshape the array into a new array where we gonna store it in a dictionary as a value on this specific key
        for key in self.dictData:
            if key == 'train' or key == 'test':
                self.dictData[key] = np.reshape(self.dictData[key], newshape=(self.dictData[key].shape[0] * self.dictData[key].shape[1], self.dictData[key].shape[2]))
            else:
                self.dictData[key] = np.reshape(self.dictData[key], newshape=(self.dictData[key].shape[0] * self.dictData[key].shape[1]))
        return self.dictData
    
    def split_Data(self):
        x_train, y_train = self.dictData['train'], self.dictData['trainlabel']
        x, val_x = np.split(x_train, [int(0.90*len(x_train))])
        y, val_y = np.split(y_train, [int(0.90*len(y_train))])
        return x, val_x, y, val_y


main = Data()
data = main.load_Data('cifar-10-batches-py')

# visualizing 
sample = data['test'][45] # taking a sample from the testing data

R = sample[0:1024].reshape(32, 32)
G = np.reshape(sample[1024:2048], newshape=(32, 32))
B = np.reshape(sample[2048:], newshape=(32, 32))
sample = np.dstack((R, G, B)) # stacking them together as a sequence depth wise (along third axis) 
plt.imshow(sample)
plt.show()

# Splitting the data into train and validation set
train_x, val_x, train_y, val_y = main.split_Data()

x_test, y_test = data['test'], data['testlabel']