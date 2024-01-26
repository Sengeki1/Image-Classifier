import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def load_Data(path):
    files = os.listdir(path=path)
    
    train = []
    trainlabel = []
    test = []
    testlabel = []

    for file in files[:6]:
        with open(path + '/' + file, 'rb') as fo: # since we are opening the files as binary
            dict = pickle.load(fo, encoding='bytes')
            if (file != 'test_batch'):
                train.append(dict[b'data']) # the data should be saved as bytes
                trainlabel.append(dict[b'labels'])
            else:
                test.append(dict[b'data'])
                testlabel.append(dict[b'labels'])
    
    print(train)

    dictData = {'train': np.array(train), 'trainlabel': np.array(trainlabel), 
                'test': np.array(test), 'testlabel': np.array(testlabel)}
    
    # We reshape the array into a new array where we gonna store it in a dictionary as a value on this specific key
    for key in dictData:
        if key == 'train' or key == 'test':
            dictData[key] = np.reshape(dictData[key], newshape=(dictData[key].shape[0] * dictData[key].shape[1], dictData[key].shape[2]))
        else:
            dictData[key] = np.reshape(dictData[key], newshape=(dictData[key].shape[0] * dictData[key].shape[1]))
    return dictData

data = load_Data('cifar-10-batches-py')

# visualizing 
temp = data['test'][45] # taking a sample from the testing data

R = temp[0:1024].reshape(32, 32)
G = np.reshape(temp[1024:2048], newshape=(32, 32))
B = np.reshape(temp[2048:], newshape=(32, 32))
temp = np.dstack((R, G, B)) # stacking them together as a sequence depth wise (along third axis) 
plt.imshow(temp)
plt.show()

# Splitting the data into train and validation set
x_train, y_train = data['train'], data['trainlabel']
x_test, y_test = data['test'], data['testlabel']

train_x, val_x = np.split(x_train, [int(0.98*len(x_train))])
train_y, val_y = np.split(y_train, [int(0.98*len(y_train))])
