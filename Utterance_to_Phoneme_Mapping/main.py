#!/usr/bin/env python
# coding: utf-8

# ## Import packages

# In[ ]:


#!cd ctcdecode && pip install .


# In[ ]:


import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import numpy as np
import time

from ctcdecode import CTCBeamDecoder
import Levenshtein as lev

from phoneme_list import N_PHONEMES, PHONEME_LIST, PHONEME_MAP
from torch.autograd import Variable


# In[ ]:


torch.cuda.empty_cache()
print(torch.__version__)


# # Hyperparameters

# In[ ]:


# data
batch_size = 64
num_workers = 4

# LSTM
nlayers = 4
dropout = 0.4

# Model
hidden_size = 512

# Training
lr = 1e-3
weight_decay = 1.2e-6
n_epochs = 50
trial = 1
main = 10

# decoder
beam_width = 30

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


# ## Dataset

# In[ ]:


class my_dataset(Dataset):
    def __init__(self, dataset, is_test=False):
        self.x = dataset[0]
        self.y = dataset[1]
        self.is_test = is_test
        
    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).float()
        if not self.is_test:   
            y = torch.from_numpy(np.array(self.y[idx]))
        else:
            y = torch.from_numpy(np.array([-1]))
        return x, y
    
    def __len__(self):
        return len(self.x)


# In[ ]:


def pad_collate(batch):
    if (batch == None):
        print("batch is none")
    X = [x[0] for x in batch] 
    Y = [x[1] for x in batch]
    padded_x = pad_sequence(X, batch_first=True) 
    padded_y = pad_sequence(Y, batch_first=True)
    x_len = torch.LongTensor([len(x) for x in X])
    y_len = torch.LongTensor([len(y) for y in Y])
    return padded_x, padded_y, x_len, y_len


# In[ ]:


PHONEME_DISTRIBUTION = np.array([0.0, 0.05358054231519048, 0.0007288214839836856, 0.013098352464491847, 0.026832112711415397, 0.09770586652641637, 0.013518714121475761, 0.0057269410467102565, 0.016955559891075733, 0.016857280892567918, 0.005520165926087382, 0.04689805694635475, 0.029505593388645272, 0.028292188327860686, 0.026502245577809964, 0.014843047952853326, 0.01762161904432918, 0.008394583367593132, 0.02126621299394319, 0.060607004178803554, 0.03551958695574964, 0.004363684839686033, 0.025278623393418128, 0.037595122637107715, 0.02825521207099636, 0.06790008431559624, 0.010302752833670579, 0.01249116340440397, 0.0010061434104661295, 0.018085281844220005, 0.03871024869938449, 0.044408484494055336, 0.007836533806759163, 0.0645148106937281, 0.006213470742293491, 0.004969900840382743, 0.012408453356154819, 0.019214030737973106, 0.022654768745137745, 0.006349212527361214, 0.027032076416300108, 0.00043544407754699514])


# In[ ]:


len(PHONEME_DISTRIBUTION)


# # Model

# In[ ]:


class locked_dropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout):
        k = 1 - dropout
        mask = x.data.new(1, x.size(1), x.size(2))
        mask = mask.bernoulli_(k)
        mask = Variable(mask, requires_grad=False) / k
        out = mask.expand_as(x) * x
        return out


# In[ ]:


class cnn_rnn_model(nn.Module):
    def __init__(self, hidden_size, nlayers, out_size=42, embed_size=40):
        super(cnn_rnn_model, self).__init__()
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.out_size = out_size
        self.embed_size = embed_size
        self.lockdrop = locked_dropout()
        
        self.cnn = torch.nn.Sequential(
            nn.Conv1d(self.embed_size, self.hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True))
        
        self.rnn = nn.LSTM( input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.nlayers,
                            bias=True,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)
        self.linear_out = nn.Linear(self.hidden_size*2, self.out_size)
        
        self.init_weights()
        
    def forward(self, x, x_lens): 
        cnn_out = self.cnn(x.permute(0, 2, 1))
        cnn_out = cnn_out.permute(2, 0, 1)
        cnn_out = self.lockdrop(cnn_out, dropout)
        rnn_in = pack_padded_sequence(cnn_out , x_lens, enforce_sorted=False)
        rnn_out, hidden = self.rnn(rnn_in)
        out, out_lens = pad_packed_sequence(rnn_out, batch_first=True)
        out = self.lockdrop(out, dropout)
        out_prob = self.linear_out(out).log_softmax(2)
        
        return out_prob.permute(1, 0, 2), x_lens
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                print("init", m)
                nn.init.kaiming_normal_(m.weight.data)
                
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.normal_(m.bias.data)
                print('initialized Linear')

            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.01)
                print('initialized BatchNorm')
                
        self.linear_out.bias.data = torch.from_numpy(PHONEME_DISTRIBUTION).float()


# In[ ]:





# In[ ]:


def decode(output_probs, data_lens, beam_width):
    decoder = CTCBeamDecoder(labels=PHONEME_MAP, 
                             beam_width=beam_width, 
                             num_processes=os.cpu_count(), 
                             log_probs_input=True)
    output_probs = torch.transpose(output_probs, 0, 1) 
    output, _, _, out_seq_len = decoder.decode(output_probs, data_lens) 
    short_list = []
    long_list = []
    for b in range(output_probs.size(0)):
        if out_seq_len[b][0] != 0:
            short_curr = "".join([PHONEME_MAP[i] for i in output[b, 0, :out_seq_len[b][0]]])
            long_curr = "".join([PHONEME_LIST[i] for i in output[b, 0, :out_seq_len[b][0]]])
        
        short_list.append(short_curr)
        long_list.append(long_curr)
    return short_list, long_list


# In[ ]:


def train_epoch(mode, data_loader, criterion, optimizer, epoch):
    model.train()
    start_time = time.time()
    
    for batch, (data, target, data_lens, target_lens) in enumerate(data_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device) 
        
        output, data_lens_new = model(data, data_lens)
        loss = criterion(output, target, data_lens_new, target_lens) 
        
        loss.backward()
        optimizer.step()
        
        end_time = time.time()
        if batch % 100 == 0:
            print("epoch:{} batch:{} time:{}".format(epoch, batch, end_time-start_time))
        
        torch.cuda.empty_cache()
        del data
        del target


# In[ ]:


def getPhonemes(target):
    return "".join([PHONEME_MAP[x] for x in target])

def calculate_levscore(w1, w2):
    return lev.distance(w1.replace(" ", ""), w2.replace(" ", ""))

def val_epoch(model, data_loader, epoch, decode_mode=False):
    with torch.no_grad():
        model.eval()
        
        start_time = time.time()
        running_loss = 0.0
        running_ls = 0.0
        total_sample_cnt = 0
        
        for batch, (data, target, data_lens, target_lens) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device) 
            output, data_lens_new = model(data, data_lens)
            loss = criterion(output, target, data_lens_new, target_lens)
            
            running_loss += loss.item()
            total_sample_cnt += len(data)
            
            if decode_mode:
                short, _ = decode(output, data_lens, beam_width)
                target_phonemes = [getPhonemes(i) for i in target]
                
                for i in range(len(target_phonemes)):
                    ls = calculate_levscore(short[i], target_phonemes[i])
                    running_ls += ls
            
            end_time = time.time()
            if batch % 100 == 0:
                print("epoch:{} batch:{} time:{}".format(epoch, batch, end_time-start_time))
            
            torch.cuda.empty_cache()
            del data
            del target
            
        loss_per_sample = running_loss / len(data_loader)
        dist_per_sample = running_ls / (len(data_loader) * 64)
        return loss_per_sample, dist_per_sample


# In[ ]:


def predict(model, data_loader):
    model.eval()
    short = np.array([])
    long = np.array([])
    start_time = time.time()
    
    for batch, (data, target, data_lens, target_lens) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        output, data_lens_new = model(data, data_lens)
        short_decode, long_decode = decode(output, data_lens_new, beam_width)
        short = np.concatenate((short, short_decode))
        long = np.concatenate((long, long_decode))
        end_time = time.time()
        print("batch:{} time{}".format(batch, end_time-start_time))
        
        torch.cuda.empty_cache()
        del data
        del target
        
    return short, long


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Load Data

# In[ ]:


train = (np.load("train.npy", allow_pickle=True), (np.load("train_labels.npy", allow_pickle=True)))
dev = (np.load("dev.npy", allow_pickle=True),(np.load("dev_labels.npy", allow_pickle=True)))
test = (np.load("test.npy", allow_pickle=True), None)


# In[ ]:


# Train
train_dataset = my_dataset(train)
train_loader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, collate_fn=pad_collate, pin_memory=True)
train_loader = DataLoader(train_dataset, **train_loader_args)
    
# Dev
dev_dataset = my_dataset(dev)
dev_loader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, collate_fn=pad_collate, pin_memory=True)
dev_loader = DataLoader(dev_dataset, **dev_loader_args)
    
# Test
test_dataset = my_dataset(test, is_test=True)
test_loader_args = dict(shuffle=False, batch_size=batch_size, num_workers=1, collate_fn=pad_collate, pin_memory=True)
test_loader = DataLoader(test_dataset, **test_loader_args)


# In[ ]:


model = cnn_rnn_model(hidden_size=hidden_size, nlayers=nlayers, out_size=42, embed_size=40)
#checkpoint = torch.load("./main7trial1/epoch25devdist8.6754.pth")
#model.load_state_dict(checkpoint["model_state_dict"])
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
criterion = nn.CTCLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3, verbose=True)
model.cuda()
print(model)


# # Train

# In[ ]:


for epoch in range(n_epochs):
    for p in optimizer.param_groups:
        print("lr:{}".format(p["lr"]))
        
    # Train
    print("**************Training epoch:{}**************".format(epoch))
    train_epoch(model, train_loader, criterion, optimizer, epoch)
    
    # Evaluate
    print("**************Evaluating epoch:{}**************".format(epoch))
    print("train")
    train_loss, train_dist = val_epoch(model, train_loader, epoch, decode_mode=False)
    print('loss: {:} distance: {:}'.format(train_loss, train_dist))
    print("val")
    dev_loss, dev_dist = val_epoch(model, dev_loader, epoch, decode_mode=True)
    print('loss: {:} distance: {:}'.format(dev_loss, dev_dist))

    scheduler.step(dev_lossPerSample)

    # Save checkpoint
    print("**************Saving**************")
    path = "./main{}trial{}/epoch{}devdist{:.4f}.pth".format(main, trial, epoch,dev_distPerSample)
    torch.save({
        "epoch":epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)
    print("="*20)


# In[ ]:





# In[ ]:





# In[ ]:





# # Predict

# In[ ]:


#checkpoint = torch.load('./main10trial1/epoch28devdist6.9817.pth')
#model.load_state_dict(checkpoint["model_state_dict"])
#model.cuda()


# In[ ]:


#short, _ = predict(model, test_loader)
#idxs = np.array(list(range(len(short))))
#df = pd.DataFrame({"id" : idxs, "label" : short})
#df.to_csv("submission.csv", index=False)


# In[ ]:


#!kaggle competitions submit -c 11785-spring2021-hw3p2 -f submission.csv -m "model 10"


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




