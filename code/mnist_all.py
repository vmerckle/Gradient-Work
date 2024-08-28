import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import torch.nn.functional as F

from mnist_utils import *



image_transform = torchvision.transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST('dataset/', train=True, download=True, transform=image_transform)
test_dataset = torchvision.datasets.MNIST('dataset/', train=False, download=True, transform=image_transform)


class NNRS(nn.Module):
    def __init__(self, m):
        super(NNRS, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(784, m),
            nn.ReLU(),
            nn.Linear(m, 1)
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc_layers(x)
        return x 

class NNR(nn.Module):
    def __init__(self, m):
        super(NNR, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(784, m),
            nn.ReLU(),
            nn.Linear(m, 10)
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc_layers(x)
        return x 

class CNNC(nn.Module):
    def __init__(self, m):
        super(CNNC, self).__init__()
        self.conv_layers = nn.Sequential(
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
            nn.Linear(320, m),
            nn.ReLU(),
            nn.Linear(m, 10)
        )

    def forward(self, x):
        # goes from 28x28 to 320
        x = self.conv_layers(x)
        # Flatten the output for fully connected layers
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        # Apply log softmax for output
        return F.log_softmax(x, dim=1)

class NNC(nn.Module):
    def __init__(self, m):
        super(NNC, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(784, m),
            nn.ReLU(),
            nn.Linear(m, 10)
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("dist", type=int)
    parser.add_argument( "-s", "--seconds", type=int, default=1)
    parser.add_argument( "-d", "--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument( "--batch_test", type=int, default=32)
    parser.add_argument( "--batch_train", type=int, default=10000)
    parser.add_argument( "-m", type=int, default=3)
    parser.add_argument( "-lr", type=float, default=1e-3)
    parser.add_argument( "--optimizer", default="adam", choices=["prod", "adam"])
    parser.add_argument( "--model", default="nnc", choices=["cnnc", "nnc", "nnrs", "nnr"])
    args = parser.parse_args()

    device = args.device
    learning_rate = args.lr
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_train, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_test, 
                                              shuffle=True)

    regression = False
    if args.model == "cnnc":
        model = CNNC(args.m) # using cpu here
        statsf = classif_stats
        loss_fn = torch.nn.NLLLoss
    elif args.model == "nnc":
        model = NNC(args.m) # using cpu here
        statsf = classif_stats
        loss_fn = torch.nn.NLLLoss
    elif args.model == "nnr":
        statsf = classif_stats_cross
        loss_fn = torch.nn.CrossEntropyLoss
        model = NNR(args.m) # using cpu here
    elif args.model == "nnrs":
        regression = True
        statsf = classif_stats_cross_scalar
        model = NNRS(args.m) # using cpu here
        loss_fn = torch.nn.MSELoss

    model = model.to(device)
    timestamp, trainlosslist, testlosslist = [], [], []

    if args.optimizer == "prod":
        from prodigyopt import Prodigy
        optimizer = Prodigy(model.parameters(), lr=1.)
    elif args.optimizer == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # example optimiser, feel free to change
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)

    timestamp, trainlosslist, testlosslist = train(model, optimizer, scheduler, loss_fn, train_loader, test_loader, device, seconds=args.seconds, regression=regression, statsf=statsf)

    # test 100% of the test data set
    test_loss, test_acc = statsf(model, device=device, loader=test_loader)
    print(f'TEST loss: {test_loss:.4f}, TEST acc: {test_acc*100:.3f}%)')
    # 100% of the train data set
    train_loss, train_acc = statsf(model, device=device, loader=train_loader)
    print(f'TRAIN loss: {train_loss:.4f}, TRAIN acc: {train_acc*100:.3f}%)')

    ts = time.time() - np.array(timestamp)
    plt.plot(timestamp, trainlosslist, label='Train Loss', color='blue')
    #plt.plot(timestamp, testlosslist, label='Test Loss(every 5 second)', color='red')
    plt.xlabel('seconds')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig("data.png", dpi=100)
