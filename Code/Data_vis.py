# **********************************************************************************************************************
# Import
# **********************************************************************************************************************
import os
import cv2
import torch
import numpy as np
import pandas as pd
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt

# **********************************************************************************************************************
# Data Prep
# **********************************************************************************************************************

Load_Path = '/home/ubuntu/Deep-Learning/Final_Project/Data_prep/'
x_train, y_train = np.load(Load_Path + "x_train.npy"), np.load(Load_Path + "y_train.npy")
x_test, y_test = np.load(Load_Path + "x_test.npy"), np.load(Load_Path + "y_test.npy")
y_label_train, y_label_test = np.load(Load_Path + "y_label_train.npy"), np.load(Load_Path + "y_label_test.npy")
# **********************************************************************************************************************
# Data Vis
# **********************************************************************************************************************

classes = np.unique(y_label_train).tolist()
#plt.imshow(x_train[5])
#plt.show()
# print labels
print("The dog breeds shown above are:" '\n',
      ', '.join('%5s' % classes[y_train[j]] for j in range(5)))

# check diversity
print("Train set contains " , len(np.unique(y_label_train)) , "categories." '\n'
      "Test  set contains " , len(np.unique(y_label_test)) , "categories.")

# **********************************************************************************************************************
# Data Frame
# **********************************************************************************************************************
# concatenate data
X, Y, Y_label = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), np.concatenate((y_label_train, y_label_test))

unique, counts = np.unique(Y_label, return_counts=True)
my_dict = dict(zip(unique, counts))

#s = pd.Series(my_dict,index=my_dict.keys())
s = pd.Series(my_dict, name='Number of Observations')
s.index.name = 'Dog Breed'
s = s.reset_index()
print(s)
s.to_csv('Dog.csv')

unique, counts = np.unique(Y, return_counts=True)
my_dict1 = dict(zip(unique, counts))
s1 = pd.Series(my_dict1, name='Number of Observations')
s1.index.name = 'Encoded Dog Breed'
s1 = s.reset_index()
#s.plot(kind='bar',x='Dog Breed',y='Number of Observations')
s1.plot(kind='bar',x='index',y='Number of Observations')
plt.savefig('plot.png')
plt.show()
