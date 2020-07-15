#!/usr/bin/env python3
## import libraries
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from scipy.stats import norm

# set GPU memory
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# specify
name = sys.argv[1]

#model_path = './saved model/500_data/prediction_not_fixed_' + name
model_path = './saved model/500_data/prediction_not_fixed_all_kind_data'
model = load_model(model_path)

dataset_path = './datasets/split by random for distributed/prediction/' + name + '_test.csv'
dataset = pd.read_csv(dataset_path, header=None)
dataset = dataset.fillna(0)
dataset = np.array(dataset)

np.set_printoptions(suppress=True)

traj = np.array([0,0,0])
error = np.array([0])
err = []
cnt = 0
flag = False
for i in range(0,250):
    tmp = dataset[i,:]
    #print(tmp.shape)
    tmp = tmp.reshape(1,tmp.shape[0])
    tmp = sequence.pad_sequences(tmp, maxlen=tmp.shape[1]+27, padding='post', dtype='float32')
    #print(tmp.reshape(int(tmp.shape[1]/3),3))
    tmp = tmp.reshape(tmp.shape[1],)
    #print(tmp.shape)
    for j in range(0, tmp.shape[0], 3):
        if tmp[j] != 0 and tmp[j+1] != 0 and tmp[j+2] != 0:
            a = tmp[j:j+3]
            #print(a)
            #print("update times ", cnt)
            if traj[0] == 0 and traj[1] == 0 and traj[2] == 0:
                traj = a
            else:
                if traj.shape[0] < 27:
                    traj = np.hstack((traj,a))
                    #print(traj.shape)
                else:
                    print("update times ", cnt)
                    if flag == False:
                        #print(traj.reshape(9,3))
                        pred = model.predict(traj.reshape(1,9,3))
                        #print("pred = \n", pred)
                        actu = tmp[j:j+27]
                        #print(actu.reshape(1,9,3))
                        flag = True
                        print('error = \n', actu.reshape(1,9,3)-pred)
                        #error = np.sum((actu.reshape(1,9,3)-pred)/27)
                        err.append(np.sum(np.power((actu.reshape(1,9,3)-pred),2))/27)
                    else:
                        traj[:-3,] = traj[3:]
                        traj[24:] = a
                        #print(traj.reshape(9,3))
                        pred = model.predict(traj.reshape(1,9,3))
                        #print("pred = \n", pred)
                        actu = tmp[j+3:j+3+27]
                        #print(actu.reshape(1,9,3))
                        print('error = \n', actu.reshape(1,9,3)-pred)
                        #error = np.hstack((error, np.sum((actu.reshape(1,9,3)-pred)/27)))
                        err.append(np.sum(np.power((actu.reshape(1,9,3)-pred),2))/27)
                        #print(np.sum(np.abs(actu.reshape(1,9,3)-pred)/27))
                    
                    cnt += 1
    #print(error)
    #print(err)
    traj = np.array([0,0,0])
    cnt = 0
    flag = False

plt.figure(figsize=(12.8,9.6))
#plt.xticks(np.arange(-int(max(err)), int(max(err)), std))
#plt.xticks(b)
#plt.xlim(-int(max(err)), int(max(err)))

df_viss = pd.DataFrame(data=err)
#data_namee = './data/tt/' + name + '_model_error_distribution.csv'
data_namee = './data/tt/' + name + '_all_data_model_error_distribution.csv'
df_viss.to_csv(data_namee, header=0, index=0)
'''
(muuu, sigmaaa) = norm.fit(err)
nss, binsss, patchesss = plt.hist(np.array(err), 100, density=True, facecolor='b', alpha=1, edgecolor='k')
print(muuu, sigmaaa)
#plt.title(r'$\mathrm{}\ \mu=%.3f,\ \sigma=%.3f$' %(muuu, sigmaaa))
plt.xlim(-5, 5)
plt.xticks(np.arange(-5, 5, 1))
yy = norm.pdf( binsss, muuu, sigmaaa)
l = plt.plot(binsss, yy, 'r', label=r'$\mathrm{}\ \mu=%.3f,\ \sigma=%.3f$' %(muuu, sigmaaa))
plt.legend(loc='upper right', prop={'size': 20})
plt.xticks(fontsize=20, fontname='FreeSerif')
plt.yticks(fontsize=20, fontname='FreeSerif')

pic_name = './' + name + '_distribution.png'
plt.savefig(pic_name)
'''
