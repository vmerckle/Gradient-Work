import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train(model, optimizer, scheduler, loss_fn, train_loader, test_loader, device, seconds=2, regression=False, statsf=None):
    timestamp, trainlosslist, testlosslist = [], [], []
    model.train()
    batch_size = train_loader.batch_size
    i = 0
    n = len(train_loader.dataset)
    samplerate, lastsample = 5, 0
    start = time.time()
    lastloss=1
    while True:
        try:
            for data, target in tqdm(train_loader, total=int(len(train_loader))):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                if regression:
                    loss = loss_fn()(output.flatten(),target.to(torch.float32).flatten()/9.-0.5)
                else:
                    loss = loss_fn()(output,target)
                loss.backward()
                optimizer.step()
                timestamp.append(time.time()-start)
                trainlosslist.append(loss.item())
                if time.time() - lastsample > samplerate:
                    print(f"{i+batch_size} samples.. {loss.item()}")
                    with torch.no_grad():
                        test_loss, test_acc = statsf(model, device=device, loader=test_loader)
                    testlosslist.append(test_loss)
                    lastsample = time.time()
                    lastloss=testlosslist[-1]
                else:
                    testlosslist.append(lastloss)
                i += batch_size
                    
                if time.time() - start > seconds:
                    break
            if time.time() - start > seconds:
                break
            print("full")
            scheduler.step()
        except KeyboardInterrupt:
            print("Normal interrupt")
    print(f" trained on {i} samples, loss = {loss.item()}, took {(time.time()-start)/60:.2f} minutes")
    return timestamp, trainlosslist, testlosslist

##define test function
def classif_stats(model, loader, device, noprint=True):
    batch_size = loader.batch_size
    model.eval()
    loss = 0
    correct = 0
    i = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += torch.nn.NLLLoss(reduction='sum')(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            i += batch_size

    loss /= i
    if not noprint:
        print(f'Average loss: {loss:.4f}, Correct/Tested: {correct}/{i} ({100. * correct / i:.0f}%)')
    accuracy = correct/i
    return loss, accuracy


def classif_stats_cross(model, loader, device, noprint=True):
    batch_size = loader.batch_size
    model.eval()
    loss = 0
    correct = 0
    i = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += torch.nn.CrossEntropyLoss(reduction='sum')(output, target).item() # sum up batch loss
            output = F.log_softmax(output, dim=0)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            i += batch_size

    loss /= i
    if not noprint:
        print(f'Average loss: {loss:.4f}, Correct/Tested: {correct}/{i} ({100. * correct / i:.0f}%)')
    accuracy = correct/i
    return loss, accuracy

def classif_stats_cross_scalar(model, loader, device, noprint=True):
    batch_size = loader.batch_size
    model.eval()
    loss = 0
    correct = 0
    i = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += torch.nn.MSELoss(reduction="sum")(output.flatten(),target.to(torch.float32).flatten()/9-0.5).item()
            correct += equals(output.flatten(), target.flatten())
            i += batch_size

    loss /= i
    if not noprint:
        print(f'Average loss: {loss:.4f}, Correct/Tested: {correct}/{i} ({100. * correct / i:.0f}%)')
    accuracy = correct/i
    return loss, accuracy


def equals(output, target):
    o = output.cpu().detach().numpy()
    # 1.3 becomes 1, 1.7 becomes 2.. -10 becomes 0
    o = np.digitize(o, np.array([-100, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 100])/9 - 0.5)-1
    o = o/9 - 0.5
    t = target.cpu().detach().numpy()
    t = t/9 - 0.5
    c = 0
    for o, t in zip(o, t):
        if o == t:
            c += 1
    return c

##define test function
def test2(model, loss_fn, loader,device="cpu", load=1, noprint=False):
    batch_size = loader.batch_size
    model.eval()
    test_loss = 0
    correct = 0
    i = 0
    n = len(loader.dataset)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.sum((output.flatten()-(target.to(torch.float32).flatten()/9-0.5))**2)
            correct += equals(output.flatten(), target.flatten())
            i += batch_size
            if i/n >= load:
                break

    test_loss /= i
    if not noprint:
        print(f'Average loss: {test_loss:.4f}, Correct/Tested: {correct}/{i} ({100. * correct / i:.0f}%)')
    return test_loss.item()/26/10
