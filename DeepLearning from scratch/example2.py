'''
Example 2: Deep Neural Network  from scrach

@Author _ Nikesh Bajaj
PhD Student at Queen Mary University of London &
University of Genova
Conact _ http://nikeshbajaj.in 
n[dot]bajaj@qmul.ac.uk
bajaj[dot]nikkey@gmail.com
'''


import numpy as np
import matplotlib.pyplot as plt
from DeepNet import deepNet
import DataSet as ds

plt.close('all')

dtype = ['MOONS','GAUSSIANS','LINEAR','SINUSOIDAL','SPIRAL']

X, y,_ = ds.create_dataset(200, dtype[3],0.0,varargin = 'PRESET');

Xts, yts,_ = ds.create_dataset(200, dtype[3],0.4,varargin = 'PRESET');

print(X.shape, y.shape)


NN = deepNet(X,y,Xts=Xts, yts=yts, Net = [8,8,5],NetAf =['tanh'], alpha=0.01,miniBatchSize = 0.3, printCostAt =100,AdamOpt=True,lambd=0,keepProb =[1.0])

plt.ion()
for i in range(15):
	NN.fit(itr=10)
	NN.PlotLCurve()
	NN.PlotBoundries(Layers=True)


NN.PlotLCurve()
NN.PlotBoundries(Layers=True)
print(NN)
yi,yp = NN.predict(X)
yti,ytp = NN.predict(Xts)
print('Accuracy::: Training :',100*np.sum(yi==y)/yi.shape[1], ' Testing :',100*np.sum(yti==yts)/yti.shape[1])