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
from utils import mdn_loss, mdn_logp



class MixtureDensityNetwork(nn.Module):
    '''Create a mixture density network
        Args:
        n_input: number of input dimensions
        n_hiddens: an integer or a list. Sizes of hidden layers.
        n_output: number of output dimensions
        n_components: number of Gaussian distributions
        logsigma_min, logsigma_max: range to clip log sigma to
        '''
    def __init__(self, n_input = 24, n_hiddens = 10, n_output = 6,
                 n_components=4,
                 logsigma_min=-3, logsigma_max=3):
        super(MixtureDensityNetwork, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_components = n_components
        self.n_hiddens = np.atleast_1d(n_hiddens)
        self.logsigma_min = logsigma_min
        self.logsigma_max = logsigma_max

        layers = []
        n_hidden_prev = n_input
        for n in self.n_hiddens:
            layers.append(nn.Linear(n_hidden_prev, n))
            layers.append(nn.ReLU())
            n_hidden_prev = n
        layers.append(nn.Linear(n_hidden_prev, n_components*(2*n_output + 1)))
        self.encoder = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        #print (self.logsigma_min)
        logsigma = torch.clamp(0.5*x[:, :self.n_components*self.n_output],
                               self.logsigma_min,
                               self.logsigma_max)

        logsigma = logsigma.view(-1, self.n_components, self.n_output)
        mu = x[:, (self.n_components*self.n_output):(self.n_components*self.n_output*2)]
        mu = mu.view(-1, self.n_components, self.n_output)
        #logpi = torch.log(self.softmax(x[:, (self.n_components*self.n_output*2):]))
        #logpi = logpi.view(-1, self.n_components)
        pi = self.softmax(x[:, (self.n_components*self.n_output*2):])
        xpi = pi+ 1e-3
        ypi= xpi/xpi.sum(dim=-1, keepdim=True)
        logpi = torch.log(ypi)
        logpi = logpi.view(-1, self.n_components)
        return logpi, logsigma, mu



class IsotropicGaussianMixture(object):
    def __init__(self, n_components=2, n_dim=1):
        self.n_components = n_components
        self.n_dim = n_dim
        self.pi = np.full(n_components, 1.0/n_components)
        self.sigma = np.ones((n_components, n_dim))
        self.mu = np.zeros((n_components, n_dim))

    def init_params(self):
        self.pi = np.random.dirichlet(alpha=[0.8]*self.n_components)
        self.mu = np.random.uniform(-3, 3, size=(self.n_components, self.n_dim))
        self.sigma = np.sqrt(np.random.gamma(shape=1, size=(self.n_components, self.n_dim)))
        return self

    def set_params(self, pi=None, mu=None, sigma=None):
        if pi is not None:
            self.pi = pi
        if mu is not None:
            self.mu = mu
        if sigma is not None:
            self.sigma = sigma
        #print (self.pi,self.mu,self.sigma)
        return self

    def pdf(self, x):
        '''Calculate probability density of given points
            Args:
            x: ndarray of shape [n_dim] or [n_samples, n_dim]
            Returns:
            p: probality densities of given points
            if x is an ndarray of shape [n_dim], p is a scalar
            if x is an ndarray of shape [n_samples, n_dim], p is a
            '''
        K, D = self.n_components, self.n_dim
        x = np.atleast_1d(x)
        if len(x.shape) == 1:
            assert x.shape[0] == D
            x = x.reshape((1, D))
            p_c = 1.0/np.power(2*np.pi, 0.5*D)/np.prod(self.sigma, axis=-1)
            p_c *= np.exp(-0.5*np.sum(np.square((x - self.mu)/self.sigma), axis=-1))
            p = np.sum(self.pi*p_c, axis=-1)
        else:
            assert x.shape[1] == D
            N = x.shape[0]
            x = x.reshape((N, 1, D))
            pi = self.pi.reshape((1, K))
            sigma = self.sigma.reshape((1, K, D))
            mu = self.mu.reshape((1, K, D))
            p_c = 1.0/np.power(2*np.pi, 0.5*D)/np.prod(sigma, axis=-1)
            p_c = p_c*np.exp(-0.5*np.sum(np.square((x - mu)/sigma), axis=-1))
            p = np.sum(pi*p_c, axis=-1)
        return p

    def __repr__(self):
        s = []
        s.append('n_components = {}'.format(self.n_components))
        s.append('n_dim = {}'.format(self.n_dim))
        s.append('pi = [{}]'.format(', '.join(self.pi.astype('str'))))
        for i in range(self.n_components):
            s.append('  mu[{:d}] = [{}]'.format(i, ', '.join(self.mu[i].astype('str'))))
            s.append('  sigma[{:d}] = [{}]'.format(i, ', '.join(self.sigma[i].astype('str'))))
        return '\n'.join(s)

    def mean_shift(self, x, tol=1e-6):
        '''Run mean-shift algorithm to find a mode
            Args:
            x: initial guess for a mode
            tol: absolute error in modes between iterations for defining convergence
            Returns:
            mode: ndarray of shape [n_dim]. Mode found.
            '''
        K, D = self.n_components, self.n_dim

        x_old = x
        while True:
            kernel = self.pi.reshape((K, 1))/np.prod(self.sigma, axis=1, keepdims=True)/np.square(self.sigma)
            kernel *= np.exp(-0.5*np.sum(np.square((x.reshape((1, D)) - self.mu)/self.sigma), axis=-1)).reshape((K, 1))
            x = np.sum(kernel*self.mu, axis=0)/np.sum(kernel, axis=0)
            if np.sqrt(np.sum(np.abs(x - x_old))) < tol:
                break
            x_old = x
        return x

    def find_modes(self, n_init=10, tol_mean_shift=1e-6, tol_merge_modes=1e-3):
        '''Find modes in the gaussian mixture model
            Args:
            n_init: number of random initializations
            tol_mean_shift: absolute error in modes between iterations for defining convergence
            tol_merge_modes: absolute error between modes for merging modes
            Returns:
            modes: ndarray of shape [n_modes, n_dim]. Modes found.
            '''
        K, D = self.n_components, self.n_dim
        # set range of high density region for finding modes
        #print self.mu, self.sigma
        range_min = np.min(self.mu - self.sigma)
        range_max = np.max(self.mu + self.sigma)
        modes = np.empty((0, D))
        for i in range(n_init):
            x = np.random.uniform(range_min, range_max, size=D)
            x_old = x
            n_iter = 0
            while True:
                n_iter += 1
                kernel = self.pi.reshape((K, 1))/np.prod(self.sigma, axis=1, keepdims=True)/np.square(self.sigma)
                kernel *= np.exp(-0.5*np.sum(np.square((x.reshape((1, D)) - self.mu)/self.sigma), axis=-1)).reshape((K, 1))
                x = np.sum(kernel*self.mu, axis=0)/(np.sum(kernel, axis=0)+10**(-4))
                if np.sum(np.abs(x - x_old)) < tol_mean_shift:
                    break
                x_old = x
            if len(modes) == 0:
                modes = np.append(modes, x.reshape((1, D)), axis=0)
            else:
                if np.min(np.sum(np.abs(x.reshape((1, D)) - modes), axis=1)) > tol_merge_modes:
                    modes = np.append(modes, x.reshape((1, D)), axis=0)
        return modes
