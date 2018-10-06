import sys,argparse,os,time
sys.path.append('bin')

parser = argparse.ArgumentParser(description='test some parameters')
parser.add_argument('--data', dest='data_name',  default='data_6', help='use different dataset')
parser.add_argument('--n_components', dest='n_components',default=6,type=int, help='number of gaussians')
parser.add_argument('--batch', dest='batchsize',default=10,type=int, help='number of samples in one batch')
parser.add_argument('--early_epoch', dest='earlystopping_epoch_nums',default=20,type=int, help='earlystopping_epoch_nums')
parser.add_argument('--nhidden1', dest='nhidden1',default=22,type=int, help='hidden1 units')
parser.add_argument('--nhidden2', dest='nhidden2',default=20,type=int, help='hidden2 units')
parser.add_argument('--nhidden3', dest='nhidden3',default=18,type=int, help='hidden3 units')
parser.add_argument('--logsigmamin', dest='logsigmamin',default=-3,type=int, help='lower bound of logsigma')
parser.add_argument('--logsigmamax', dest='logsigmamax',default=3,type=int, help='upper bound of logsigma')
args = parser.parse_args()

dir_name = ('batch_'+str(args.batchsize)+'_gaussian_'+str(args.n_components)+
            '_datatype_'+args.data_name+'_batch_'+str(args.batchsize)+'_earlyepoch_'+str(args.earlystopping_epoch_nums)
            +'_nhidden_1_'+str(args.nhidden1)+'_logsigma_min_max_'+str(args.logsigmamin)
            +'_'+str(args.logsigmamax))
path_model = os.path.join('models_new/', dir_name)

if os.path.exists(path_model+'/paras'):
    print ('parameter combinations tested, skip')
    sys.exit()




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
from tqdm import tqdm

from utils import prepare_dataset, preprocess,preprocess_zscore, get_original_parameters, report_metrics, mdn_logp, mdn_loss
from model import MixtureDensityNetwork, IsotropicGaussianMixture



X_train, X_test, y_train, y_test = prepare_dataset(featurename=args.data_name)
print (args.logsigmamin)

model = MixtureDensityNetwork(n_input=X_train.shape[1], n_output=2,  n_components = args.n_components,
                              n_hiddens=[args.nhidden1,args.nhidden2, args.nhidden3],
                              logsigma_min=args.logsigmamin, logsigma_max=args.logsigmamax)
optimizer = optim.Adam(model.parameters())

# create data loaders
scalers = {}
datas = [X_train, X_test, y_train, y_test]
for i in range(4):
    datas[i],scalers[i]  = preprocess_zscore(datas[i])
X_train_, X_test_, y_train_, y_test_ = datas
batch_size = args.batchsize
train_ = utilsdata.TensorDataset(torch.from_numpy(X_train_.astype('float32')),
                                 torch.from_numpy(y_train_.astype('float32')))
test_ = utilsdata.TensorDataset(torch.from_numpy(X_test_.astype('float32')),
                                torch.from_numpy(y_test_.astype('float32')))
train_loader_ = torch.utils.data.DataLoader(
                                            dataset=train_,
                                            batch_size=batch_size,
                                            shuffle=True)
test_loader_ = torch.utils.data.DataLoader(
                                           dataset=test_,
                                           batch_size=batch_size,
                                           shuffle=False)

print('X_train.shape =', X_train.shape, 'X_test.shape =', X_test.shape,
      'y_train.shape =', y_train.shape, 'y_test.shape =', y_test.shape)


trainlosses, testlosses = {},{}
for epoch in tqdm(range(2000)):
    train_loss = []
    for i_batch, batch_data in enumerate(train_loader_):
        x, y = batch_data
        model.zero_grad()
        logpi, logsigma, mu = model(x)
        loss = mdn_loss(y, logpi, logsigma, mu)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item()*x.size()[0])
    train_loss = np.sum(train_loss)/len(train_loader_.dataset)

    test_loss = []
    with torch.no_grad():
        for i_batch, batch_data in enumerate(test_loader_):
            x, y = batch_data
            logpi, logsigma, mu = model(x)
            loss = mdn_loss(y, logpi, logsigma, mu)
            test_loss.append(loss.item()*x.size()[0])
        test_loss = np.sum(test_loss)/len(test_loader_.dataset)
    trainlosses[epoch] = train_loss
    testlosses[epoch] = test_loss
    if epoch%10 == 0:
        print('[Epoch {:d}] train loss: {}, test loss: {}'.format(epoch, train_loss, test_loss))
    ###### early stop to avoid unnecessary training######
    if epoch >=200:
        if epoch%10 == 0:
            recentlossmin = np.min(np.array([testlosses[i] for i in np.arange(epoch-args.earlystopping_epoch_nums,epoch)]))
            otherlossmin = np.min(np.array([testlosses[i] for i in np.arange(0,epoch-args.earlystopping_epoch_nums)]))
            print (recentlossmin,otherlossmin)
            if recentlossmin > otherlossmin: # no longer decrease
                print ('exist at epoch:' +str(epoch))
                break


if not os.path.exists(path_model):
    os.makedirs(path_model)
torch.save(model.state_dict(), path_model+'/model')


logpi_pred, logsigma_pred, mu_pred = model(torch.Tensor(X_test_))
pi_reversed, sigma_reversed, mu_reversed = get_original_parameters(logpi_pred, logsigma_pred, mu_pred)

def get_prediction(pi,mu,sigma,n_components):
    model = IsotropicGaussianMixture(n_components, n_dim=2)
    model.set_params(pi,mu,sigma)
    modes = model.find_modes(n_init=10)
    p_modes = model.pdf(modes)
    #print p_modes,modes
    index=np.where(p_modes==np.max(p_modes))
    return p_modes[index[0]], modes[index[0]]

prediction_xy = np.ndarray([250,2])
probabes = np.ndarray([250])
for i in tqdm(range(250)):
    probabes[i], prediction_xy[i] = get_prediction(pi_reversed[i], mu_reversed[i],sigma_reversed[i],args.n_components)
prediction_xy_reverse = scalers[3].inverse_transform(prediction_xy)


parametersinfo = np.array([args.data_name,str(args.n_components),str(args.batchsize), str(args.earlystopping_epoch_nums),str(args.nhidden1),str(args.nhidden2),str(args.nhidden3),str(args.logsigmamin),str(args.logsigmamax)])

if not os.path.exists(path_model+'/paras'):
    os.makedirs(path_model+'/paras')

np.savetxt(path_model+'/paras/rmse.txt',np.array([report_metrics(prediction_xy_reverse,y_test)[0]]))
np.savetxt(path_model+'/paras/pcc.txt',np.array([report_metrics(prediction_xy_reverse,y_test)[1][0]]))
np.savetxt(path_model+'/paras/probabse.txt',probabes)
np.savetxt(path_model+'/paras/parameters.txt',parametersinfo,fmt="%s")



