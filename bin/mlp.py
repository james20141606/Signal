import gc, argparse, sys, os, errno
import numpy as np
import pandas as pd
import h5py
import os
import scipy
import sklearn
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
import numpy as np # basic math and random numbers
import torch # package for building functions with learnable parameters
import torch.nn as nn # prebuilt functions specific to neural networks
import torch.utils.data
from torch.autograd import Variable # storing data while learning
from scipy import stats
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils import data as utilsdata
import torch.nn.functional as F

def prepare_dataset(featurename='rss'):
    import scipy.io as sio
    data_1 = sio.loadmat('data_paper_WSNL/1000data.mat')
    data_2 = sio.loadmat('data_paper_WSNL/TXRXLoc2000.mat')
    local = data_2['RXm'][:1000,:2]
    rss = data_1['data_db_rss']
    aoa = data_1['data_db_aoa']
    toa = data_1['data_db_toa']
    data_4 = data_1['data'][:,18:24]
    data_5 = data_1['data'][:,24:]
    data_6 = data_1['data'][:,6:]
    data_whole = np.concatenate((rss,aoa,toa,data_4,data_5),axis =1)
    local_x = local[:,:1]
    local_y = local[:,1:]
    if featurename=='whole':
        return train_test_split(data_whole, local, random_state=42)
    elif featurename=='rss':
        return train_test_split(rss, local, random_state=42)
    elif featurename=='aoa':
        return train_test_split(aoa, local, random_state=42)
    elif featurename=='toa':
        return train_test_split(toa, local, random_state=42)
    elif featurename=='data_4':
        return train_test_split(data_4, local, random_state=42)
    elif featurename=='data_5':
        return train_test_split(data_5, local, random_state=42)
    elif featurename=='data_6':
        return train_test_split(data_6, local, random_state=42)
X_train, X_test, y_train, y_test = prepare_dataset(featurename='data_6')

# Dataloader
batch_size = 5
train = utilsdata.TensorDataset(torch.from_numpy(X_train.astype('float32')), torch.from_numpy(y_train.astype('float32')))
test = utilsdata.TensorDataset(torch.from_numpy(X_test.astype('float32')), torch.from_numpy(y_test.astype('float32')))
train_loader = torch.utils.data.DataLoader(
                                           dataset=train,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(
                                          dataset=test,
                                          batch_size=batch_size,
                                          shuffle=False)

## network
class MLPNet(nn.Module):
    def __init__(self,n_input = 24,n_hidden_1 = 18,n_hidden_2 = 12,n_hidden_3 = 6,n_output = 2,n_samples=750,epoch_num=100000):
        super(MLPNet, self).__init__()
        self.n_input  = n_input
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_output =  n_output
        self.fc1 = nn.Linear(n_input, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_hidden_3, n_output)
    def forward(self, x):
        x = x.view(-1, self.n_input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def name(self):
        return "MLP"
use_cuda = True
model =  MLPNet()
if use_cuda:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


epochnum = 100000
print ('epochnum'+str(epochnum))
aveloss = np.ndarray([epochnum])
for epoch in range(epochnum):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        #print (x.size(),target.size())
        out = model(x)
        loss = criterion(out, target)
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            print ('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                                                                                 epoch, batch_idx+1, ave_loss))
    aveloss[epoch] = ave_loss
np.savetxt('loss7.27.txt',aveloss)
torch.save(model.state_dict(), 'MLP.7.27')

