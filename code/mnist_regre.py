import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from mnist_utils import *

import torchvision
import torch

hiddenlayer_neurons = 10
batch_size_train = 32*4*2*2*2*2*2*2*2*2 # We use a small batch size here for training
batch_size_train = 60000  # We use a small batch size here for training
batch_size_test = 10000 #
print(batch_size_train)
device="cuda"

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = torchvision.datasets.CIFAR10('dataset/', train=True, download=True, transform=image_transform)
test_dataset = torchvision.datasets.CIFAR10('dataset/', train=False, download=True, transform=image_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size_train, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size_test, 
                                          shuffle=True)
for d,t in train_loader:
    print(d.shape)
    print(t.shape)

for d,t in test_loader:
    print(d.shape)
    print(t.shape)

import sys
sys.exit(0)
import torch.nn.functional as F

m = hiddenlayer_neurons

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(784, m),
            nn.ReLU(),
            nn.Linear(m, 1)
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc_layers(x)
        return x 


cnn_model = CNN() # using cpu here
cnn_model = cnn_model.to(device)
timestamp, trainlosslist, testlosslist = [], [], []
elapsed = 0
learning_rate = 1e-4
#learning_rate = 1.
#from prodigyopt import Prodigy
#optimizer = Prodigy(cnn_model.parameters(), lr=learning_rate) # example optimiser, feel free to change
optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=learning_rate) # example optimiser, feel free to change
loss_fn = torch.nn.MSELoss()

def train2(model, seconds=2, fromwhere=0):
    timestamp, trainlosslist, testlosslist = [], [], []
    model.train()
    batch_size = train_loader.batch_size
    i = 0
    n = len(train_loader.dataset)
    samplerate, lastsample = 1, 0
    start = time.time()
    lastloss=1
    while True:
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            print(data.shape)
            optimizer.zero_grad()
            output = cnn_model(data)
            loss = loss_fn(output.flatten(),target.to(torch.float32).flatten()/9-0.5)
            loss.backward()
            optimizer.step()

            timestamp.append(time.time()-start+elapsed)
            trainlosslist.append(loss.item())
            if time.time() - lastsample > samplerate:
                testlosslist.append(test2(cnn_model, device=device, loss_fn=loss_fn, loader=test_loader, load=0.05, noprint=True))
                model.train()
                print(".", end="", flush=True)
                lastsample = time.time()
                lastloss=testlosslist[-1]
            else:
                testlosslist.append(lastloss)
            i += batch_size
                
            if time.time() - start > seconds:
                break
        print("full")
        if time.time() - start > seconds:
            break
    print(f" trained on {i} samples, loss = {loss.item()}")
    return timestamp, trainlosslist, testlosslist, time.time() - start



def train3(model, seconds=2, fromwhere=0):
    for d,t in train_loader:
        fullb, fullt = d.to(device), t.to(device)
    timestamp, trainlosslist, testlosslist = [], [], []
    model.train()
    batch_size = train_loader.batch_size
    i = 0
    n = len(train_loader.dataset)
    samplerate, lastsample = 1, 0
    start = time.time()
    lastloss=1
    while True:
        if True:
        #for data, target in train_loader:
            #data, target = data.to(device), target.to(device)
            data, target = fullb, fullt
            optimizer.zero_grad()
            output = cnn_model(data)
            loss = loss_fn(output.flatten(),target.to(torch.float32).flatten()/9-0.5)
            loss.backward()
            optimizer.step()

            timestamp.append(time.time()-start+elapsed)
            trainlosslist.append(loss.item())
            testlosslist.append(lastloss)
            i += batch_size
                
            if time.time() - start > seconds:
                break
        print(loss.item())
        if time.time() - start > seconds:
            break
    print(f" trained on {i} samples, loss = {loss.item()}")
    return timestamp, trainlosslist, testlosslist, time.time() - start

a,b,c, e = train3(cnn_model, seconds=10, fromwhere=elapsed) # train for 20 seconds
timestamp.extend(a)
trainlosslist.extend(b)
testlosslist.extend(c)
elapsed += e
test2(cnn_model, device=device, loss_fn=loss_fn, loader=test_loader, load=0.1) # test 100% of the test data set
ts = time.time() - np.array(timestamp)
plt.plot(timestamp, trainlosslist, label='Train Loss', color='blue')
#plt.plot(timestamp, testlosslist, label='Test Loss(every 5 second)', color='red')
plt.xlabel('seconds')
plt.ylabel('Loss')
plt.title('Train and Test Loss Over Time')
plt.legend()
plt.grid(True)
plt.savefig("data.png", dpi=100)
