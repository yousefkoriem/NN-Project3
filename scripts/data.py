import os
from keras.utils import to_categorical
from keras.datasets import cifar10
import numpy as np
import pickle

print(os.getcwd())
(x_train, y_train), (x_test,y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train,num_classes=10)
y_test = to_categorical(y_test,num_classes=10)
pickle.dump(obj=(x_train,y_train),file=open('data/processed/train_data.pkl', 'wb'))
pickle.dump(obj=(x_test,y_test),file=open('data/processed/test_data.pkl', 'wb'))