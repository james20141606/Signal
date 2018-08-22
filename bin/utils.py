import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable # storing data while learning
from torch import optim
from torch.utils import data as utilsdata
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from scipy.signal import argrelmax
import scipy
import numpy as np

def mdn_logp(x, logpi, logsigma, mu):
    '''Loss function of a mixture density network is the negative log likelihood of a Gaussian mixture
        Args:
        x: Tensor of shape [batch_size, n_dim]
        logpi: Tensor of shape [batch_size, n_components]
        logsigma: Tensor of shape [batch_size, n_components, n_dim]
        mu: Tensor of shape [batch_size, n_components, n_dim]
        Returns:
        Log likelihoods of input samples. Tensor of shape [batch_size]
        '''
    batch_size, n_components, n_dim = logsigma.size()
    x = x.view(batch_size, -1, n_dim)
    logpi = logpi.view(batch_size, n_components, -1)
    var = torch.pow(torch.exp(logsigma), 2)
    #print(x.size(), logpi.size(), logsigma.size(), mu.size())
    ll_gaussian = -float(0.5*np.log(2*np.pi)) - logsigma - 0.5/var*torch.pow(x - mu, 2)
    ll = torch.logsumexp(ll_gaussian + logpi, 1)
    return ll

def mdn_loss(x, logpi, logsigma, mu):
    '''Same as mdn_logp except that the log likelihoods are negated and averaged across samples
        Returns:
        Negative log likelihood of input samples averaged over samples. A scalar.
        '''
    return torch.mean(-mdn_logp(x, logpi, logsigma, mu))

def prepare_dataset(featurename='rss'):
    import scipy.io as sio
    data_1 = sio.loadmat('data_paper_WSNL/1000data.mat')
    data_2 = sio.loadmat('data_paper_WSNL/location.mat')
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


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def preprocess(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data),scaler

def preprocess_zscore(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data),scaler

def get_original_parameters(logpi, logsigma, mu):
    '''
        input scaled and logged
        output exped, not reversed yet
        '''
    pi = np.exp(logpi.detach().numpy())
    sigma = np.exp(logsigma.detach().numpy())
    mu = mu.detach().numpy()
    return pi, sigma, mu

def report_metrics(y_test_data,y_test):
    rmse = np.mean(np.sum((y_test_data - y_test)**2,axis=1)**0.5)
    pcc = scipy.stats.pearsonr(y_test_data.ravel(),y_test.ravel())
    return rmse,pcc
