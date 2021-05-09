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
      
context = 15
num_workers = 0

class MyTestDataset(Dataset):
    def __init__(self, X, context):
        self.X = X
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
    
        return X
    
test_data = np.load('test_data_stack.npy', allow_pickle=True)
test_dataset = MyTestDataset(test_data, context)
test_loader_args = dict(shuffle=False, batch_size=128, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=False, batch_size=1)
test_loader = DataLoader(test_dataset, **test_loader_args)

input_layer = [(2 * context + 1) * 40]
output_layer = [71]
hidden_layers = [1024,1024,1024,1024,512,256]

model_test = Simple_MLP(input_layer + hidden_layers + output_layer)
model_test.load_state_dict(torch.load("trial5epoch6acc75.66240361962392.pth"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_test.parameters(), )
device = torch.device("cuda" if cuda else "cpu")
model_test.to(device)
print(model_test)

def Finaltest_model(model, ftest_loader, criterion):
    print("Final Testing....")
    with torch.no_grad():
        model.eval()

        p = np.array([])
        i = 0
        for batch_idx, (data) in enumerate(ftest_loader):
            data = data.to(device)
            outputs = model(data)

            predicted = torch.argmax(outputs.data, 1)
            p = np.concatenate((p, predicted.cpu().numpy().reshape(-1)))
        
        return p
      
p = Finaltest_model(model, test_loader, criterion)
print("Done")

labels = list(map(int, p))
ids = np.array(list(range(len(labels))))
label = np.array(labels)

df = pd.DataFrame({"id" : ids, "label" : label})
df.to_csv("results.csv", index=False)


