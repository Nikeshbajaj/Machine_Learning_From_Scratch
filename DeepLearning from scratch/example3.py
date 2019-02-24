'''
Example 3: Deep Neural Network  from scrach

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

Xi, yi = ds.mclassGaus(N=500, nClasses = 4,var =0.25,ShowPlot=False)

[n,N] =Xi.shape

r  = np.random.permutation(N)

X = Xi[:,r[:N//2]]
y = yi[:,r[:N//2]]
Xts =Xi[:,r[N//2:]]
yts =yi[:,r[N//2:]]

print(X.shape, y.shape,Xts.shape,yts.shape)

NN = deepNet(X,y,Xts=Xts, yts=yts, Net = [8,8,5],NetAf =['tanh'], alpha=0.01,miniBatchSize = 0.3, printCostAt =-1,AdamOpt=True,lambd=0,keepProb =[1.0])

plt.ion()
for i in range(10):
	NN.fit(itr=10)
	NN.PlotLCurve()
	NN.PlotBoundries(Layers=True)


NN.PlotLCurve()
NN.PlotBoundries(Layers=True)
print(NN)
yi,yp = NN.predict(X)
yti,ytp = NN.predict(Xts)
print('Accuracy::: Training :',100*np.sum(yi==y)/yi.shape[1], ' Testing :',100*np.sum(yti==yts)/yti.shape[1])