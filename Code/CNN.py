# --------------------------------------------------------------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from torch.autograd import Variable

# **********************************************************************************************************************
# Set-Up
# **********************************************************************************************************************
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# --------------------------------------------------------------------------------------------
# Hyper Parameters
input_size = 3* 100 *100
hidden_size = 500
num_classes = 120
num_epochs = 10
batch_size = 64
learning_rate = 0.001
SIZE_IMG = 100
# --------------------------------------------------------------------------------------------

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# --------------------------------------------------------------------------------------------
# **********************************************************************************************************************
# Data Preparation
# **********************************************************************************************************************
# # Reshapes to (n_examples, n_channels, height_pixels, width_pixels)
Load_Path = '/home/ubuntu/Deep-Learning/Final_Project/Data_prep/'

x_train, y_train = np.load(Load_Path + "x_train.npy", allow_pickle=True), np.load(Load_Path + "y_train.npy", allow_pickle=True)
x_train, y_train = torch.from_numpy(x_train).float().to(device), torch.from_numpy(y_train).float().to(device)
x_train = x_train.data.view(len(x_train), 3, SIZE_IMG, SIZE_IMG).float().to(device)

x_train.requires_grad = True

x_test, y_test = np.load(Load_Path + "x_test.npy"), np.load(Load_Path + "y_test.npy")
x_test, y_test = torch.from_numpy(x_test).float().to(device), torch.from_numpy(y_test).float().to(device)
x_test, y_test = x_test.data.view(len(x_test), 3, SIZE_IMG, SIZE_IMG).float().to(device), y_test.to(device)

y_label_train, y_label_test = np.load(Load_Path + "y_label_train.npy"), np.load(Load_Path + "y_label_test.npy")

train_set = data_utils.TensorDataset(x_train.detach().to(torch.device("cpu")), y_train.long().to(torch.device("cpu")))
train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = data_utils.TensorDataset(x_test.detach().to(torch.device("cpu")), y_test.long().to(torch.device("cpu")))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

classes = np.unique(y_label_train).tolist()
# --------------------------------------------------------------------------------------------

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    img = img.to(torch.device("cpu"))
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()


# show images

imshow(torchvision.utils.make_grid(images))
plt.show()
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# --------------------------------------------------------------------------------------------
# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
# --------------------------------------------------------------------------------------------

net = Net(input_size, hidden_size, num_classes)
net.cuda()
# --------------------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# --------------------------------------------------------------------------------------------
# Train the Model
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        # Convert torch tensor to Variable
        images, labels = data
        images = images.view(-1, 3 * 100 * 100).cuda()

        # wrap them in Variable
        images, labels = Variable(images), Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.data[0]))
# --------------------------------------------------------------------------------------------
# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 3 * 100 * 100)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
# --------------------------------------------------------------------------------------------

_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
# --------------------------------------------------------------------------------------------


class_correct = list(0. for i in range(120))
class_total = list(0. for i in range(120))
for data in train_loader:
    images, labels = data
    images = Variable(images.view(-1, 3* 100 * 100)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    labels = labels.cpu().numpy()
    c = (predicted.cpu().numpy() == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

# --------------------------------------------------------------------------------------------
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

# --------------------------------------------------------------------------------------------
# Save the Model
torch.save(net.state_dict(), 'model.pkl')