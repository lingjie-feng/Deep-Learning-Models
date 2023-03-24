import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
import matplotlib.pyplot as plt

# global variables
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
sys.version
print(cuda, sys.version)

class MyDataset(Dataset):
    def __init__(self, X, Y, context):
        self.X = X
        self.Y = Y 
        self.context = context
        self.len = len(self.X)
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        k = self.context
        
        if (index <= k):
            diff = k - index
            X = self.X[:(index+k+1)]
            X = np.pad(X, ((diff, 0), (0, 0)), 'constant', constant_values=(0, 0))
            X = torch.tensor(X.reshape(-1)).float()
        elif ((index + k) >= self.len):
            diff = index + k + 1 - self.len
            X = self.X[(index - k):self.len]
            X = np.pad(X, ((0, diff), (0, 0)), 'constant', constant_values=(0, 0))
            X = torch.tensor(X.reshape(-1)).float()
        else: 
            X = self.X[(index - k):(index+k+1)]
            X = torch.tensor(X.reshape(-1)).float()
        
        Y = torch.tensor(self.Y[index]).long()
        
        return X, Y
      
      
context = 15
num_workers = 8 if cuda else 0

# Load Data
train_data = np.load('train_data_stack.npy', allow_pickle=True)
train_labels = np.load('train_labels_stack.npy', allow_pickle=True)

train_dataset = MyDataset(train_data, train_labels, context)
train_loader_args = dict(shuffle=True, batch_size=128, num_workers=num_workers, pin_memory=True, drop_last=True) if cuda else dict(shuffle=True, batch_size=64)
train_loader = DataLoader(train_dataset, **train_loader_args)

dev_data = np.load('dev_data_stack.npy', allow_pickle=True)
dev_labels = np.load('dev_labels_stack.npy', allow_pickle=True)

dev_dataset = MyDataset(dev_data, dev_labels, context)
dev_loader_args = dict(shuffle=False, batch_size=128, num_workers=1, pin_memory=True) if cuda                        else dict(shuffle=False, batch_size=1)
dev_loader = DataLoader(dev_dataset, **dev_loader_args) 

class Simple_MLP(nn.Module): 
    def __init__(self, size_list):
        super(Simple_MLP, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2): 
            if i < 5: 
                layers.append(nn.Linear(size_list[i], size_list[i+1]))
                layers.append(nn.BatchNorm1d(size_list[i+1]))
                layers.append(nn.ReLU())
            else: 
                layers.append(nn.Linear(size_list[i], size_list[i+1]))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers) 
    
    def forward(self, x): 
        return self.net(x)
      
# Create Model
input_layer = [(2 * context + 1) * 40]
output_layer = [71]
hidden_layers = [1024,1024,1024,1024,512,256]
model = Simple_MLP(input_layer + hidden_layers + output_layer)
model.to(device)
print(model)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001) 

def train_epoch(model, train_loader, criterion, optimizer): 
    print("Training...")
    model.train() 
    
    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0

    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device) 
        
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
            
        predicted = torch.argmax(outputs.data, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()
        
    
    
    end_time = time.time()

    running_loss /= len(train_loader)
    print("Training Loss: ", running_loss, "Time: ", end_time - start_time, "s")
    acc = (correct_predictions/total_predictions)*100.0
    print("Training Accuracy: ", acc, "%")
    return running_loss, acc
  
def val_model(model, val_loader, criterion):
    print("Validating....")
    with torch.no_grad():
        model.eval() 

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.reshape(-1)
            target = target.to(device)
            
            outputs = model(data) 

            predicted = torch.argmax(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()

    end_time = time.time()

    running_loss /= len(val_loader)
    print("Evaluation Loss: ", running_loss, "Time: ", end_time - start_time, "s")
    acc = (correct_predictions/total_predictions)*100.0
    print("Evaluation Accuracy: ", acc, "%")
    return running_loss, acc
  
Train_loss = []
Train_acc = []
Val_loss = []
Val_acc = []
trial_number = 6
n_epochs = 10

for i in range(n_epochs):
        print("Epoch: ", i)
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        end_time = time.time()
        
        start_time = time.time()
        val_loss, val_acc = val_model(model, dev_loader, criterion)
        end_time = time.time()
        scheduler.step(val_loss)
        
        Train_loss.append(train_loss)
        Train_acc.append(train_acc)
        Val_loss.append(val_loss)
        Val_acc.append(val_acc) 
        
        filename = "trial"+str(trial_number)+"epoch"+str(epoch)+"acc"+str(val_acc)+".pth"
        torch.save(model.state_dict(), filename)
        print('='*20)
