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

# Load data
raw_train = np.load('train.npy', allow_pickle=True)
raw_train_labels = np.load('train_labels.npy', allow_pickle=True)
print("raw_train shape: ", raw_train.shape)
print("raw_train_labels shape: ", raw_train_labels.shape)

raw_dev = np.load('dev.npy', allow_pickle=True)
raw_dev_labels = np.load('dev_labels.npy', allow_pickle=True)
print("raw_dev shape: ", raw_dev.shape)
print("raw_dev_labels shape: ", raw_dev_labels.shape)

train_data = np.vstack(raw_train) 
print("train_data shape: ", train_data.shape)
np.save("train_data_stack.npy", train_data)
train_labels = np.hstack(raw_train_labels)
print("train_labels shape: ", train_labels.shape)
np.save("train_labels_stack.npy", train_labels)

dev_data = np.vstack(raw_dev) 
print("dev_data shape: ", dev_data.shape)
np.save("dev_data_stack.npy", train_data)
dev_labels = np.hstack(raw_dev_labels)
print("dev_labels shape: ", dev_labels.shape)
np.save("dev_labels_stack.npy", train_data)

raw_test = np.load('test.npy', allow_pickle=True)
print("raw_test shape: ", raw_test.shape)
test_data = np.vstack(raw_test) 
print("test_data shape: ", test_data.shape)
np.save("test_data_stack.npy", train_data)
