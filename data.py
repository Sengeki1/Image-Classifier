import os
import pickle

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

    
    dictData = {}
    


data = load_Data('cifar-10-batches-py')