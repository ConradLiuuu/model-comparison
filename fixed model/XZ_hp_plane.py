#!/usr/bin/env python3
## import libraries
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

# set GPU memory
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

np.set_printoptions(suppress=True)

# specify
name = sys.argv[1]

model_path = './saved model/500_data/prediction_fixed_' + name
model = load_model(model_path)

dataset_path = './datasets/split by random for distributed/prediction/' + name + '_test.csv'
dataset = pd.read_csv(dataset_path, header=None)
dataset = dataset.fillna(0)
dataset = np.array(dataset)

def calculate_vis_hitting_point(arr):
    __hitting_point = -45
    if (-50 <= float(np.min(arr)) <= -45):
        index = np.argmin(arr)
        row = int((index-0)/3)
        w1 = (arr[row,1]-__hitting_point) / (arr[row,1]-arr[row-1,1])
        __vis_hitting_point = (1-w1)*arr[row,:] + w1*arr[row-1,:]
    elif (-45 < float(np.min(arr)) < -40):
        index = np.argmin(arr)
        row = int((index-0)/3)
        w1 = (arr[row,1]-__hitting_point) / (arr[row-1,1]-arr[row,1])
        __vis_hitting_point = arr[row,:] - w1*(arr[row-1,:]-arr[row,:])
    return __vis_hitting_point

def modify_input(arr):
    ka = np.array([0])
    for i in range(arr.shape[0]):
        if arr[i,] != 0:
            ka = np.hstack((ka,arr[i,]))
    ka = ka[1:,]
    return ka

hitting_point = -45
__hp = -45
err = np.zeros((1,3))
flag = True

for i in range(dataset.shape[0]):
    tmp = dataset[i,:] ## read trajectory
    ip = modify_input(tmp) ## remove zeros from pandas
    
    ## calculate vis hitting point
    ipp = ip.reshape(int(ip.shape[0]/3),3)
    if (-50 <= float(np.min(ipp)) <= -45):
        index = np.argmin(ipp)
        row = int((index-0)/3)
        w1 = (ipp[row,1]-hitting_point) / (ipp[row,1]-ipp[row-1,1])
        vis_hitting_point = (1-w1)*ipp[row,:] + w1*ipp[row-1,:]
    elif (-45 < float(np.min(ipp)) < -40):
        index = np.argmin(ipp)
        row = int((index-0)/3)
        w1 = (ipp[row,1]-hitting_point) / (ipp[row-1,1]-ipp[row,1])
        vis_hitting_point = ipp[row,:] - w1*(ipp[row-1,:]-ipp[row,:])
    print(vis_hitting_point)
    
    ## padding for model
    ip2 = sequence.pad_sequences(ip.reshape(1,ip.shape[0]), maxlen=ip.shape[0]+27, padding='post', dtype='float32')
    
    for i in range(0,(ip2.shape[1]-27),3):
        t = ip2[:,i:i+27]
        pred = model.predict(t.reshape(1,9,3))
        #print("input = \n", t.reshape(1,9,3))
        #print("output = \n", pred)
        if -55 < pred[0,0,1] < -45 and flag == True:
            #print(t[0,-3:])
            w = abs(__hp-t[0,-3:]) / abs(pred[0,0,1]-t[0,-3:])
            hp = w*pred[0,0,:] + (1-w)*t[0,-3:].reshape(1,3)
            print("pred hp a = \n", hp)
            flag = False
            #input()
        if -45 < pred[0,0,1] < -40 and flag == True:
            w = abs(__hp-pred[0,0,1]) / (pred[0,0,1]-t[0,-2])
            hp = pred[0,0,:] - w*(pred[0,0,:]-t[0,-3:])
            print("pred hp b = \n", hp)
            flag = False
            #input()
    flag = True
    error = vis_hitting_point - hp
    err = np.vstack((err, error))

r = 7.5
err2 = err[1:,:]
correct = 0
for i in range(err2.shape[0]):
    if err2[i,0] < r and err2[i,0] > -r and err2[i,2] < r and err2[i,2] > -r:
        correct += 1
print(correct)
print("accuray = {}%".format(correct/err2.shape[0]*100))
text = str("Probability = {}%".format(np.round(correct/err2.shape[0]*100,2)))

fig_path_svg = './pic/XZ_plane/'+name+'_test.svg'
fig_path_png = './pic/XZ_plane/'+name+'_test.png'

plt.figure(figsize=(8,8))
x = y = np.arange(-r, r, 0.005)
x, y = np.meshgrid(x,y)
#plt.contour(x, y, x**2 + y**2, [15*15])
circle2 = plt.Circle((0, 0), r, color='r', fill=False)

ax = plt.gca()
ax.add_artist(circle2)
ax.set_xlim((-2*r, 2*r))
ax.set_ylim((-2*r, 2*r))

plt.scatter(err2[:,0], err2[:,2], edgecolors='b', marker='x', color='b')
#plt.scatter(0,0, color='r', marker='o')

#plt.ylim(-20,20)
plt.xlabel('X coordinate error (cm)', fontsize=24, fontname='FreeSerif')
plt.ylabel('Z coordinate error (cm)', fontsize=24, fontname='FreeSerif')
plt.xticks(fontsize=18, fontname='FreeSerif')
plt.yticks(fontsize=18, fontname='FreeSerif')
plt.grid(True)
plt.text(0,(2*r-2.5),text, fontsize=24, fontname='FreeSerif', horizontalalignment='center', verticalalignment='center')
plt.tight_layout(pad=2.5)
plt.title(name+' X Z hitting plane', fontsize=24, fontname='FreeSerif', fontweight='bold')
plt.savefig(fig_path_svg)
plt.savefig(fig_path_png)
