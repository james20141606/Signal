#! /usr/bin/env python
import os, sys
import numpy as np
from tqdm import tqdm
import math
rmsess = np.loadtxt('rmses_.txt',dtype='str')
indunder200 = np.where(rmsess[:,1].astype('float') <=200)[0]
save_path = 'bestparas_new_ran/'
model_path = 'models_new/'

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
sys.path.append('bin')
from utils import mdn_loss, mdn_logp

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from utils import prepare_dataset, preprocess,preprocess_zscore, get_original_parameters, report_metrics, mdn_logp, mdn_loss
from model import MixtureDensityNetwork, IsotropicGaussianMixture

import matplotlib.pyplot as plt

def plot_result(ax,ind,pi_reversed,mu_reversed,sigma_reversed,n_components=20):
    #print (n_components)
    gaussianmodel = IsotropicGaussianMixture(n_components, n_dim=2)
    gaussianmodel.set_params(pi_reversed[ind],  mu_reversed[ind], sigma_reversed[ind])
    X_grid, Y_grid = np.mgrid[-3:3:0.02, -3:3:0.02]
    X = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    modes = gaussianmodel.find_modes(n_init=10)
    #print ('modes',modes)
    print ('p_modes',gaussianmodel.pdf(modes))
    ax.contour(X_grid, Y_grid, gaussianmodel.pdf(X).reshape(X_grid.shape),
               linewidths=0.5, levels=np.linspace(0, 1, 80))
    ax.set_title('sample '+str(ind))

def get_prediction(pi,mu,sigma,n_components):
    model = IsotropicGaussianMixture(n_components, n_dim=2)
    model.set_params(pi,mu,sigma)
    modes = model.find_modes(n_init=10)
    p_modes = model.pdf(modes)
    #print p_modes,modes
    index=np.where(p_modes==np.max(p_modes))
    return p_modes[index[0]], modes[index[0]]
def get_prediction_random(pi, mu, sigma, n_components):
    modes=np.zeros(2)
    for i in range(n_components):
        cov=[[sigma[i][0],0],[0,sigma[i][1]]]
        s = np.random.multivariate_normal(mu[i], cov).T
        modes=modes+pi[i]*s
    return modes
    

def get_probability(pred_xy,pi,mu,sigma,n_components):
    l=200
    real_xy=scalers[3].inverse_transform(pred_xy)
    real_xy_u=real_xy+l/2
    real_xy_l=real_xy-l/2
    prediction_xy_u=scalers[3].transform(real_xy_u)[0]
    prediction_xy_l=scalers[3].transform(real_xy_l)[0]
    p=0
    for i in range(n_components):
        x_u=scipy.special.erf((prediction_xy_u[0]-mu[i][0])/(sqrt(2)*sigma[i][0]))
        x_l=scipy.special.erf((prediction_xy_l[0]-mu[i][0])/(sqrt(2)*sigma[i][0]))
        y_u=scipy.special.erf((prediction_xy_u[1]-mu[i][1])/(sqrt(2)*sigma[i][1]))
        y_l=scipy.special.erf((prediction_xy_l[1]-mu[i][1])/(sqrt(2)*sigma[i][1]))
        p=p+pi[i]*0.25*(x_u-x_l)*(y_u-y_l)
    return p

def plot_arrow(y_testdata,y_test):
    fig,ax=plt.subplots(1,figsize=(9,9))
    c = np.sum(y_testdata**2+y_test**2,axis=1)
    ax.scatter(y_testdata[:,0],y_testdata[:,1], s=25, c=c, cmap=plt.cm.coolwarm, zorder=10)
    ax.scatter(y_test[:,0],y_test[:,1], s=25, c=c, cmap=plt.cm.coolwarm, zorder=10)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]
            #ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    xorigin = y_testdata[:,0].ravel()
    yorigin = y_testdata[:,1].ravel()
    dx = y_test[:,0].ravel()-y_testdata[:,0].ravel()
    dy = y_test[:,1].ravel()-y_testdata[:,1].ravel()
    print (xorigin.shape, yorigin.shape, dx.shape, dy.shape)
    ax.arrow(xorigin[2], yorigin[2], dx[2], dy[2], color = 'b')
    for i in range(y_testdata.shape[0]):
        ax.arrow(y_testdata[i,0],y_testdata[i,1],
                 y_test[i,0]-y_testdata[i,0],y_test[i,1]-y_testdata[i,1],color = 'b')




class visualize_report(object):
    def __init__(self, indneed):
        self.indneed = indneed
        self.bestparas = np.loadtxt(model_path+rmsess[indneed][0]+'/paras/parameters.txt',dtype='str')
        X_train, X_test, y_train, self.y_test = prepare_dataset(featurename=self.bestparas[0])
        #print("Size of features in training data: {}".format(X_train.shape))
        #print("Size of output in training data: {}".format(y_train.shape))
        #print("Size of features in test data: {}".format(X_test.shape))
        #print("Size of output in test data: {}".format(y_test.shape))
        scalers = {}
        datas = [X_train, X_test, y_train, self.y_test]
        for i in range(4):
            datas[i],scalers[i]  = preprocess_zscore(datas[i])
        X_train_, X_test_, y_train_, y_test_ = datas
        model = MixtureDensityNetwork(n_input=X_train.shape[1], n_output=2,  n_components = int(self.bestparas[1]),n_hiddens=[int(self.bestparas[4]), int(self.bestparas[4])-2, int(self.bestparas[4])-4],logsigma_min=int(self.bestparas[-2]), logsigma_max=int(self.bestparas[-1]))
        model.load_state_dict(torch.load(model_path+rmsess[indneed][0]+'/model'))
        logpi_pred, logsigma_pred, mu_pred = model(torch.Tensor(X_test_))
        #logpi_pred.size(), logsigma_pred.size(), mu_pred.size()
        self.pi_reversed, self.sigma_reversed, self.mu_reversed = get_original_parameters(logpi_pred, logsigma_pred, mu_pred)
        #print (self.pi_reversed.shape, self.sigma_reversed.shape, self.mu_reversed.shape)
        self.prediction_xy = np.ndarray([250,2])
        self.prediction_xy_random = np.ndarray([250,2])
        self.probabes = np.ndarray([250])
        for i in tqdm(range(250)):
            _, pred = get_prediction(self.pi_reversed[i], self.mu_reversed[i],self.sigma_reversed[i],int(self.bestparas[1]))
            self.probabes[i]=get_probability(pred,self.pi_reversed[i],self.mu_reversed[i],self.sigma_reversed[i],int(self.bestparas[1])
            #self.prediction_xy[i]=pred
            self.prediction_xy_random[i]=get_prediction_random(self.pi_reversed[i], self.mu_reversed[i], self.sigma_reversed[i],int(self.bestparas[1]))
        self.prediction_xy_reverse = scalers[3].inverse_transform(self.prediction_xy)
        self.prediction_xy_random_reverse = scalers[3].inverse_transform(self.prediction_xy_random)
        self.rmseall = np.sum((self.prediction_xy_reverse-self.y_test)**2,axis=1)**0.5
        self.rmseall_random = np.sum((self.prediction_xy_random_reverse-self.y_test)**2,axis=1)**0.5

        scaler = StandardScaler()
        self.rmseall_random_ = scaler.fit_transform(self.rmseall_random.reshape(-1,1)).ravel()
        self.probabes_ = scaler.fit_transform(self.probabes.reshape(-1,1)).ravel()
    def plot_distribution(self,save=False):
        fig,ax=plt.subplots(4,4,figsize=(20,20))
        for i in range(4):
            for j in range(4):
                plot_result(ax[i,j],10+i*4+j,self.pi_reversed,self.mu_reversed,self.sigma_reversed,int(self.bestparas[1]))
        if save ==True:
            if not os.path.exists(save_path+rmsess[self.indneed ][0]):
                os.makedirs(save_path+rmsess[self.indneed ][0])
            fig.savefig(save_path+rmsess[self.indneed ][0]+'/distribution.png')   # save the figure to file
            plt.close(fig)
    def plot_arrows(self):
        plot_arrow(self.prediction_xy_reverse,self.y_test)
    def plot_correlation(self):
        fig,ax=plt.subplots(1,figsize=(20,5))
        ax.plot(self.probabes_)
        ax.plot(-self.rmseall_)
    def report(self,save=False):
        print('rmse-random', (np.mean(self.rmseall)-np.mean(self.rmseall_ramdom)))
        #print ('RMSE, PCC: ',report_metrics(self.prediction_xy_reverse,self.y_test))
        #print ('correlation of RMSE and probes: ',scipy.stats.pearsonr(self.probabes_,-self.rmseall_))
        if save:
            if not os.path.exists(save_path+rmsess[self.indneed][0]):
                os.makedirs(save_path+rmsess[self.indneed ][0])
            np.savetxt(save_path+rmsess[self.indneed][0]+'/RMSE_PCC.txt',np.array([ report_metrics(self.prediction_xy_reverse,self.y_test)[0],scipy.stats.pearsonr(self.probabes_,-self.rmseall_)[0] ]))
            np.savetxt(save_path+rmsess[self.indneed][0]+'/RMSE_sample.txt',self.rmseall)
            np.savetxt(save_path+rmsess[self.indneed][0]+'/pmodes_sample.txt',self.probabes)
            np.savetxt(save_path+rmsess[self.indneed][0]+'/pi_sample.txt',self.pi_reversed)





for i in range(indunder200.shape[0]):
    if os.path.exists(save_path+rmsess[indunder200[i]][0]):
        print ('parameter combinations tested, skip')
    else:
        visual = visualize_report(indunder200[i])
        visual.report(True)
        visual.plot_distribution(True)        
print(os.path)




