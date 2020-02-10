# **********************************************************************************************************************
# Import
# **********************************************************************************************************************
import os
import cv2
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# **********************************************************************************************************************
# Load Data
# **********************************************************************************************************************

if "Images" not in os.listdir():
    os.system("cd ~/Project/")
    os.system("wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar")
    os.system("tar -xvf images.tar")

os.listdir("Images")

DATA_DIR = os.getcwd() + "/Images/"
RESIZE_TO = 100, 100
x, y = [], []
for i in range(len(os.listdir(DATA_DIR))):
    for path in [f for f in os.listdir(DATA_DIR+os.listdir(DATA_DIR)[i])]:
        x.append(cv2.resize(cv2.imread(DATA_DIR + os.listdir(DATA_DIR)[i] + '/' + path), (RESIZE_TO)))
        label = os.listdir(DATA_DIR)[i]
        y.append(label)

x, y = np.array(x), np.array(y)
y_label = y

# **********************************************************************************************************************
# One-Hot-Encode
# **********************************************************************************************************************

# integer encode
le = preprocessing.LabelEncoder()
le.fit(os.listdir(DATA_DIR))
list(le.classes_)
y = le.transform(y)
print(x.shape, y.shape)


# **********************************************************************************************************************
# Data Split
# **********************************************************************************************************************

x_train, x_test, y_train, y_test, y_label_train, y_label_test = train_test_split(x, y, y_label, random_state=1, test_size=0.3)
np.save("x_train.npy", x_train); np.save("y_train.npy", y_train); np.save("y_label_train.npy", y_label_train);
np.save("x_test.npy", x_test); np.save("y_test.npy", y_test); np.save("y_label_test.npy", y_label_test)

# **********************************************************************************************************************
# Data Vis
# **********************************************************************************************************************

classes = os.listdir(DATA_DIR)
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

imshow(torchvision.utils.make_grid(torch.tensor(x_train[:5]).view(len(x_train[:5]), 3, 100, 100)))
plt.show()
# print labels
print("The dog breeds shown above are:" '\n',
      ', '.join('%5s' % classes[y_train[j]] for j in range(5)))

# check diversity
print("Train set contains " , len(np.unique(y_label_train)) , "categories." '\n'
      "Test  set contains " , len(np.unique(y_label_test)) , "categories.")