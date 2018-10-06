#! /usr/bin/env python
import os
import numpy as np
from tqdm import tqdm
rmsesss = []
names = []
for i in tqdm(os.listdir('models_new/')):
    try:
        #        print (np.loadtxt('models/'+i+'/paras/rmse.txt'))
        rmsesss.append(np.loadtxt('models_new/'+i+'/paras/rmse.txt').ravel()[0])
        names.append(i)
    except:
        print ('no such file: '+i)
#print (np.sort(rmsesss))
save = np.ndarray([len(rmsesss),2]).astype('object')
for i in tqdm(range(len(rmsesss))):
    save[i,0] = names[i]
    save[i,1] = str(rmsesss[i])
np.savetxt('rmses_.txt',save,fmt='%s')
print ('tested condition nums: '+str((len(os.listdir('models_new/')))) )
print ("minimum:  "+str(np.min(rmsesss)))
