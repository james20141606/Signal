#! /usr/bin/env python

import gc, argparse, sys, os, errno
import numpy as np
import os
from tqdm import tqdm
import scipy
from scipy.stats import pearsonr
from scipy.io import loadmat
from matplotlib.mlab import griddata
tableau20 = np.array([(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)])/255.
styles = ["white","dark",'whitegrid',"darkgrid"]
contexts = ['paper','talk','poster','notebook']
from ipywidgets import interact, FloatSlider,IntSlider, RadioButtons,Dropdown,Tab,Text,SelectMultiple,FloatRangeSlider
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d.axes3d import *
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler,Imputer
from sklearn.feature_selection import VarianceThreshold
sys.path.append('bin')
from utils import prepare_dataset,preprocess_zscore, get_original_parameters, report_metrics, mdn_logp, mdn_loss
from model import MixtureDensityNetwork, IsotropicGaussianMixture
import argparse
import numba as nb

parser = argparse.ArgumentParser(description='test some correlation')
parser.add_argument('--c', dest='count',  default=10,type=int, help='feature counts')
parser.add_argument('--t', dest='test_count', default=1000,type=int, help='feature test counts')
parser.add_argument('--p', dest='index_count', default=1000,type=int,help='point test counts')
args = parser.parse_args()
feature_counts = args.count
point_counts = args.index_count
test_counts = args.test_count
rawdata = loadmat('/home/xupeng/projects/signal/data/citydata.mat')
RSS = rawdata['RSS']
TOA = rawdata['TOA']
DOA = rawdata['DOA']
TXloc = rawdata['TX']
RXloc = rawdata['RX']
xyzorigin_RX = np.array([np.min(RXloc[:,0]),np.min(RXloc[:,1]),0])
xyzorigin_TX = np.array([np.min(TXloc[:,0]),np.min(TXloc[:,1]),0])
RXloc_shifted = RXloc - xyzorigin_RX
TXloc_shifted = TXloc - xyzorigin_TX

distance=np.loadtxt('/home/xupeng/projects/signal/citydata/distance.txt',dtype=np.float32)
#distance=np.zeros((1500,1500))
#for i in tqdm(range(1500)):
#    for j in range(1500):
#        distance[i,j]=np.linalg.norm(RXloc_shifted[i]-RXloc_shifted[j])

c=np.append(RSS,TOA,axis=1)
combine_data=np.append(c,DOA,axis=1)
#@nb.jit(nopython=True)
def pcc(X, Y):

    X -= np.mean(X)
    Y -= np.mean(Y)
    num=np.sqrt(np.sum(X*X)*np.sum(Y*Y))

    return np.sum(X*Y)/num
#@nb.jit(nopython=True)
def relate(index_1, index_2, features):
    xdata=combine_data[index_1]
    xdata=xdata[features]
    ydata=combine_data[index_2]
    ydata=ydata[features]
    an=~np.isnan(xdata)&~np.isnan(ydata)
    xdata=xdata[an]
    nr=1-len(xdata)/len(ydata)
    if len(xdata) == 0:
        return np.nan,1
    ydata=ydata[an]
    an=an[an]
    r=scipy.stats.pearsonr(xdata,ydata)[0]
    return r,nr
#@nb.jit(nopython=True)
def corelate(features):
    batchsize=100;
    r=np.zeros(testnum)
    nr=np.zeros(testnum)
    xn=np.random.randint(0,1500,size=(testnum,batchsize))
    yn=np.random.randint(0,1500,size=(testnum,batchsize))
    r=np.zeros((testnum,batchsize))
    u=np.zeros((testnum,batchsize))
    d=np.zeros((testnum,batchsize))
    re=np.zeros(testnum)
    nr=np.zeros(testnum)
    for j in range(testnum):
        nan_num=0;
        for i in range(batchsize):
            d[j][i]=distance[xn[j][i],yn[j][i]]
            r[j][i],u[j][i]=relate(xn[j][i],yn[j][i],features)
            while np.isnan(r[j][i]):
                xn[j][i]=np.random.randint(0,1500)
                yn[j][i]=np.random.randint(0,1500)
                d[j][i]=distance[xn[j][i],yn[j][i]]
                r[j][i],u[j][i]=relate(xn[j][i],yn[j][i],features)
                nan_num=nan_num+1
        re[j]=scipy.stats.pearsonr(d[j],r[j])[0]
        nr[j]=(np.sum(u[j])+nan_num)/(nan_num+batchsize)
    return -np.mean(re), np.mean(nr), xn.flatten(), yn.flatten(), r.flatten(), u.flatten()

batchsize=100;
testnum=int(point_counts/batchsize)
x_total=np.zeros((test_counts,testnum*batchsize))
y_total=np.zeros((test_counts,testnum*batchsize))
r_total=np.zeros((test_counts,testnum*batchsize))
u_total=np.zeros((test_counts,testnum*batchsize))
relation=np.zeros(test_counts)

label=np.random.randint(0,500,size=(test_counts,feature_counts))

nan_rate=np.zeros(test_counts)

for i in tqdm(range(test_counts)):
    relation[i],nan_rate[i],x_total[i],y_total[i],r_total[i],u_total[i]=corelate(label[i])


index=np.argsort(-relation)
relation=relation[index]
label=label[index]
nan_rate=nan_rate[index]
x_total=x_total[index]
y_total=y_total[index]
u_total=u_total[index]
r_total=r_total[index]

dir_name=(str(feature_counts)+'feature_counts'+str(test_counts)+'test_counts'+str(point_counts)+'point_counts')
path_model = os.path.join('/home/xupeng/projects/signal/citydata/', dir_name)
if not os.path.exists(path_model):
    os.makedirs(path_model)


np.savetxt(path_model+'/pcc.txt',relation)
np.savetxt(path_model+'/label.txt',label,fmt='%d')
np.savetxt(path_model+'/nan_rate.txt',nan_rate,fmt='%4.3f')
np.savetxt(path_model+'/first_index.txt',x_total,fmt='%d')
np.savetxt(path_model+'/second_index.txt',y_total,fmt='%d')
np.savetxt(path_model+'/nanrate_specific.txt',u_total,fmt='%3.2f')
np.savetxt(path_model+'/relation_specific.txt',r_total)