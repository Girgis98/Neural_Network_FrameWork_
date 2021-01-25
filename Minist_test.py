import numpy as n
import pandas as pd
from Utils import *
from Layers import *
from Layers import *
from Data import *
from Model import *


train_path = 'mnist_train_small.csv'
test_path = 'mnist_test.csv'
img_data = Image(path = train_path,test_frac = 0.8,val_frac = 0.195,image_size=(28,28), colour = 'gray')
img_data_test = Image(path = test_path,test_frac = 0.8,val_frac = 0.195,image_size=(28,28), colour = 'gray')
train_data = img_data.train_data
test_data = img_data_test.test_data

print(np.amax(train_data))
print(np.amax(test_data))


weights = np.random.random((10,784))
bias = np.random.random((10,1))

train_data = np.reshape(train_data,(100,1,784))
train_data = np.reshape(train_data,(100,784))
train_data = np.reshape(train_data,(784,100))
test_data = np.reshape(test_data,(784,7999))
test_data = test_data[:,0:100]

train_labels = img_data.train_labels
test_labels = img_data.test_labels

#train_labels = np.reshape(train_labels,(1,100))
train_labels = ((np.random.random((1,100))) * 2).astype(int)
train_labels = np.random.randint(0,7,(1,100),int)
test_labels = np.reshape(test_labels,(1,15999))
test_labels = test_labels[:,0:100]
#test_labels = np.random.randint(0,8,(1,100),int)
test_labels = np.copy(train_labels) + np.random.randint(0,2,(1,100),int)

Layer1 = Linear(train_data, weights, bias, 'softmax')
my_Model = Model([Layer1])


# Layer1()
my_Loss = my_Model.Loss(train_labels, Layer1.output()).CrossEntropy()
opt = my_Model.Optimization(100, my_Loss, 0.1, 0.02,'Batch').RMSprop()
my_Model.Evaluate(train_data, train_labels, True)
#my_Model.Evaluate(test_data, test_labels, True)
model_utls = Model_Utils()

model_utls.save(my_Model,'Minist_with_random_labels')

