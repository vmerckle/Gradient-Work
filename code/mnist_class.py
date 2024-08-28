import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from mnist_utils import *

import torchvision
import torch
import torch.nn.functional as F

hiddenlayer_neurons = 2
batch_size_train = 32
batch_size_test = 100
learning_rate = 1e-3
device="cuda"
device="cpu"

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = torchvision.datasets.MNIST('dataset/', train=True, download=True, transform=image_transform)
test_dataset = torchvision.datasets.MNIST('dataset/', train=False, download=True, transform=image_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size_train, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size_test, 
                                          shuffle=True)

m = hiddenlayer_neurons

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            # Convolutional layer
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            nn.ReLU(),
            # Max pooling layer with kernel size 2x2
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
            nn.ReLU(),
            #nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Define fully connected layers
        self.fc_layers = nn.Sequential(
            # Fully connected layer with 320 input features and m output features
            nn.Linear(320, m),
            nn.ReLU(),
            #nn.Dropout(),
            # Fully connected layer with m input features and 10 output features, the 10 classes.
            nn.Linear(m, 10)
        )

    def forward(self, x):
        # Forward pass through convolutional layers
        # goes from 28x28 to 320
        x = self.conv_layers(x)
        # Flatten the output for fully connected layers
        x = x.view(-1, 320)
        # Forward pass through fully connected layers
        x = self.fc_layers(x)
        # Apply log softmax for output
        return F.log_softmax(x, dim=1)

cnn_model = CNN() # using cpu here
cnn_model = cnn_model.to(device)
timestamp, trainlosslist, testlosslist = [], [], []
elapsed = 0

#learning_rate = 1.
#from prodigyopt import Prodigy
#optimizer = Prodigy(cnn_model.parameters(), lr=learning_rate) # example optimiser, feel free to change
optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=learning_rate) # example optimiser, feel free to change
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)
loss_fn = F.nll_loss

def train(model, seconds=2, fromwhere=0):
    timestamp, trainlosslist, testlosslist = [], [], []
    model.train()
    batch_size = train_loader.batch_size
    i = 0
    n = len(train_loader.dataset)
    samplerate, lastsample = 5, 0
    start = time.time()
    lastloss=1
    while True:
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = cnn_model(data)
            loss = loss_fn(output,target)
            loss.backward()
            optimizer.step()
            timestamp.append(time.time()-start+elapsed)
            trainlosslist.append(loss.item())
            if time.time() - lastsample > samplerate:
                print(f"{i+batch_size} samples.. {loss.item()}")
                testlosslist.append(test(cnn_model, device=device, loss_fn=loss_fn, loader=test_loader, load=0.05))
                model.train()
                lastsample = time.time()
                lastloss=testlosslist[-1]
            else:
                testlosslist.append(lastloss)
            i += batch_size
                
            if time.time() - start > seconds:
                break
        print("full")
        scheduler.step()
        if time.time() - start > seconds:
            break
    print(f" trained on {i} samples, loss = {loss.item()}")
    return timestamp, trainlosslist, testlosslist, time.time() - start

a,b,c, e = train(cnn_model, seconds=10, fromwhere=elapsed) # train for 20 seconds
timestamp.extend(a)
trainlosslist.extend(b)
testlosslist.extend(c)
elapsed += e
test(cnn_model, device=device, loss_fn=loss_fn, loader=test_loader, load=0.1) # test 100% of the test data set

ts = time.time() - np.array(timestamp)
plt.plot(timestamp, trainlosslist, label='Train Loss', color='blue')
#plt.plot(timestamp, testlosslist, label='Test Loss(every 5 second)', color='red')
plt.xlabel('seconds')
plt.ylabel('Loss')
plt.title('Train and Test Loss Over Time')
plt.legend()
plt.grid(True)
plt.savefig("data.png", dpi=100)
