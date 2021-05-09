#!/usr/bin/env python
# coding: utf-8

# # HW2P2 Bootcamp
# ___
# 
# * Custom Dataset & DataLoader
# * Torchvision ImageFolder Dataset
# * Residual Block
# * CNN model with Residual Block
# * Training 
# * Cosine Similarity
# * Center Loss
# * Triplet Loss

# ## Imports

# In[ ]:


import os
import numpy as np
import pandas as pd
import torch
import torchvision   
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
import re
from sklearn.metrics import roc_auc_score


# In[ ]:


torch.cuda.empty_cache()


# # Hyperparameters

# In[ ]:


learning_rate = 0.1
weight_decay = 5e-5
batch_size = 128
trial_number = 24
n_epochs = 70


# In[ ]:





# In[ ]:





# In[ ]:





# ## Residual Block
# 
# Resnet: https://arxiv.org/pdf/1512.03385.pdf

# In[ ]:


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


# In[ ]:


# Basic residual block 
class resnet_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,):
        super(resnet_block, self).__init__()
        self.downsample = downsample
        self.relu = nn.ReLU()

        self.layers = nn.Sequential(
            conv3x3(in_channel, out_channel, stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel)
        )
        
    def forward(self, x):
        if self.downsample is None:
            identity = x
        else: 
            identity = self.downsample(x)
            
        out = self.layers(x)
        out = self.relu(out+identity)

        return out


# In[ ]:


# Resnet model
# Modification to the original resnet model: reduce the kernal size from 7 to 1, stride from 2 to 1, and padding 
# from 3 to 1 in the first layer; added embedding
# Used Kaiming initialization for weights
class resnet(nn.Module):

    def __init__(self, block, layers, in_features, num_classes, feat_dim=2):
        super(resnet, self).__init__()
        self.in_channel = 64
        self.conv2d = nn.Conv2d(in_features, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(self.in_channel)
        
        nn.init.kaiming_normal_(self.conv2d.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.batchnorm.weight, 1)
        nn.init.constant_(self.batchnorm.bias, 0)
        
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        
        self.layers = nn.Sequential(
            self.conv2d,
            self.batchnorm,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
        )
        
        self.linear = nn.Linear(512, feat_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear_output = nn.Linear(512,num_classes)  

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x) 
        output = self.linear_output(embedding)
        if return_embedding:
            return embedding,output
        else:
            return output
    
    def make_layer(self, block, out_channel, num_blocks, stride=1):
        if stride != 1:
            downsample = nn.Sequential(conv1x1(self.in_channel, out_channel, stride), nn.BatchNorm2d(out_channel))
        else: 
            downsample = None
            
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        for i in range(1, num_blocks):
            layers.append(block(out_channel, out_channel))
        
        self.in_channel = out_channel
        return nn.Sequential(*layers)


# In[ ]:


def resnet18(num_classes):
    return resnet(resnet_block, [2, 2, 2, 2], 3, num_classes)

def resnet34(num_classes):
    return resnet(resnet_block, [3, 4, 6, 3], 3, num_classes)


# In[ ]:





# # Get data

# In[ ]:


# Data augmentation: random flip, random rotation, ad color jitter
data_transform = transforms.Compose([                           
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2),
    transforms.ToTensor()])

data_transform_val = transforms.Compose([                                                   
    transforms.ToTensor()])


# In[ ]:


train_dataset = torchvision.datasets.ImageFolder(root='train_data', transform=data_transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

dev_dataset = torchvision.datasets.ImageFolder(root='val_data', transform=data_transform_val)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)


# # Train Model

# In[ ]:


num_classes = len(train_dataset.classes)
print(num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = resnet34(num_classes)
# uncomment the following line if you want to load the state of a model previously trained
#network.load_state_dict(torch.load("./trial24/epoch5acc78.2875.pth"))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=3, verbose=True)

network = network.to(device)
print(network)


# In[ ]:


# Use scaler to speed up the training 
def train_epoch(model, train_loader, criterion, optimizer): 
    print("Training...")
    model.train() 
    
    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    
    scaler = torch.cuda.amp.GradScaler()
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device) 
        
        with torch.cuda.amp.autocast():
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        
        scaler.update()
        
        predicted = torch.argmax(outputs.data, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()
        
    end_time = time.time()

    running_loss /= len(train_loader)
    print("Training Loss: ", running_loss, "Time: ", end_time - start_time, "s")
    acc = (correct_predictions/total_predictions)*100.0
    print("Training Accuracy: ", acc, "%")
    return running_loss, acc


# In[ ]:


def val_model(model, val_loader, criterion):
    print("Validating....")
    with torch.no_grad():
        model.eval() 

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)
            
            outputs = model(data) 

            predicted = torch.argmax(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()
            
            torch.cuda.empty_cache()
            del data
            del target
        
    end_time = time.time()

    running_loss /= len(val_loader)
    print("Evaluation Loss: ", running_loss, "Time: ", end_time - start_time, "s")
    acc = (correct_predictions/total_predictions)*100.0
    print("Evaluation Accuracy: ", acc, "%")
    return running_loss, acc


# In[ ]:


Train_loss = []
Train_acc = []
Val_loss = []
Val_acc = []


# In[ ]:


for epoch in range(n_epochs):
        print("Epoch: ", epoch)
        start_time = time.time()
        train_loss, train_acc = train_epoch(network, train_dataloader, criterion, optimizer)
        end_time = time.time()
        
        start_time = time.time()
        val_loss, val_acc = val_model(network, dev_dataloader, criterion)
        end_time = time.time()
        scheduler.step(val_loss)
        
        Train_loss.append(train_loss)
        Train_acc.append(train_acc)
        Val_loss.append(val_loss)
        Val_acc.append(val_acc) 
        
        directory_name = "./trial"+str(trial_number) + "/"
        filename = "epoch"+str(epoch)+"acc"+str(val_acc)+".pth"
        torch.save(network.state_dict(), directory_name+filename)
        print('='*40)


# In[ ]:





# In[ ]:


plt.plot(Train_acc)
plt.plot(Val_acc)


# In[ ]:





# # Get test dataset

# In[ ]:


def parse_data(data_dir):
    img_list = []
    for root, directories, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(".jpg"):
                filei = os.path.join(root, filename)
                img_list.append(filei)
    print("{} # Images".format(len(img_list)))
    return img_list


# In[ ]:


class my_dataset_test(Dataset):
    def __init__(self, file_list, n_class):
        self.file_list = file_list
        self.n_class = n_class

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        return img, self.file_list[index]


# In[ ]:


img_list = parse_data('test_data')
img_list.sort(key= lambda x: int(re.split('/|\.', x)[1]))
img_list[:10] #see if test images are sorted by their names correctly 


# In[ ]:


test_dataset = my_dataset_test(img_list, 4000)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=1, drop_last=False)


# In[ ]:





# In[ ]:





# In[ ]:





# # Classify Test data

# ##### Go back to the cell that creates and loads the model. Change the file path in the line that starts with "network.load_state_dict" to the model with the best performance (highest accuracy) so far. 

# In[ ]:


def test_label(model, test_loader, criterion):
    print("Final Testing....")
    with torch.no_grad():
        model.eval()

        p = np.array([])
        for batch_idx, (data) in enumerate(test_loader):
            data = data[0]
            data = data.to(device)
            outputs = model(data)
            predicted = torch.argmax(outputs.data, 1)
            p = np.concatenate((p, predicted.cpu().numpy().reshape(-1)))
            
            torch.cuda.empty_cache()
            del data
        
        return p


# In[ ]:


p = test_label(network, test_loader, criterion)
print("Done")
p = [int(x) for x in p]


# In[ ]:


labels = list(map(int, p))
label = np.array(labels)
ids = np.loadtxt('classification_test.txt', dtype=str)


# In[ ]:


# mapping to true labels
train_labels = train_dataset.class_to_idx
lbs = dict((v,k) for k,v in train_labels.items())
label = np.array(list(map(lbs.get, label)))


# In[ ]:


# output the prediction 
df = pd.DataFrame({"id" : ids, "label" : label})
df.to_csv("results.csv", index=False)


# In[ ]:





# In[ ]:





# # Calculate verification score

# In[ ]:


class my_dataset_verify(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        triplet = self.file_list[index].split()
        imgA = Image.open("../hw2p2s2/" + triplet[0])
        imgA = transforms.ToTensor()(imgA)
        imgB = Image.open("../hw2p2s2/" + triplet[1])
        imgB = transforms.ToTensor()(imgB)
        
        if len(triplet) == 3: 
            return imgA, imgB, int(triplet[2])
        else: 
            return imgA, imgB, -1


# In[ ]:


def parse_data_verify(data_pair):
    img_list = []
    with open(data_pair) as f: 
        for line in f: 
            img_list.append(line)
            
    print("{} # Images".format(len(img_list)))
    return img_list


# In[ ]:


verify_list = parse_data_verify('../hw2p2s2/verification_pairs_val.txt')
verify_dataset = my_dataset_verify(verify_list)
verify_dataloader = DataLoader(verify_dataset, shuffle=False, batch_size=1, num_workers=1, pin_memory=True)


# In[ ]:





# In[ ]:


def verify_val(model, dataLoader):
    similarity = np.array([])
    actual_label = np.array([])
    compute_sim = nn.CosineSimilarity(dim=0)
    
    with torch.no_grad():
        for i, (imgA, imgB, label) in enumerate(dataLoader):
            imgA, imgB, label = imgA.to(device), imgB.to(device), label.to(device)
            imgA_out = model(imgA, return_embedding=True)[0].squeeze(0)
            imgB_out = model(imgB, return_embedding=True)[0].squeeze(0)
            sim = compute_sim(imgA_out, imgB_out) 
                
            similarity = np.concatenate((similarity, sim.cpu().numpy().reshape(-1)))
            actual_label = np.concatenate((actual_label, label.cpu().numpy().reshape(-1)))
            if i % 100 == 0:
                print("Batch: {}\t".format(i))
            
            torch.cuda.empty_cache()
            del imgA
            del imgB
            del label
            
    return similarity, actual_label


# In[ ]:


# calculate similarity score
similarity, actual_label = verify_val(network, verify_dataloader)
print(similarity.__len__())
print(actual_label.__len__())


# In[ ]:


auc = roc_auc_score(actual_label, similarity)
print(auc)


# In[ ]:





# In[ ]:


verify_test_list = parse_data_verify('../hw2p2s2/verification_pairs_test.txt')
verify_test_dataset = my_dataset_verify(verify_test_list)
verify_test_dataloader = DataLoader(verify_test_dataset, shuffle=False, batch_size=1, num_workers=1, pin_memory=True)


# In[ ]:


similarity_test, _ = verify_val(network, verify_test_dataloader)


# In[ ]:


# output the result
ids = np.array(verify_list_test)
df = pd.DataFrame({"Id" : ids, "Category" : similarity_test})
df.to_csv("verify_result.csv", index=False)

