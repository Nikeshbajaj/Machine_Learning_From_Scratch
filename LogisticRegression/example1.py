'''
Machine Learning from scrach
Example 1: Logistic Regression

@Author _ Nikesh Bajaj
PhD Student at Queen Mary University of London &
University of Genova
Conact _ http://nikeshbajaj.in 
n[dot]bajaj@qmul.ac.uk
bajaj[dot]nikkey@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import DataSet as ds
from LogisticRegression import LR

np.random.seed(1)
plt.close('all')

dtype = ['MOONS','GAUSSIANS','LINEAR','SINUSOIDAL','SPIRAL']

X, y,_ = ds.create_dataset(200, dtype[3],0.05,varargin = 'PRESET')


print(X.shape, y.shape)

means = np.mean(X,1).reshape(X.shape[0],-1)
stds  = np.std(X,1).reshape(X.shape[0],-1)

X = (X-means)/stds


Clf = LR(X,y,alpha=0.003,polyfit=True,degree=5,lambd=2)

plt.ion()
fig=plt.figure(figsize=(8,4))
gs=GridSpec(1,2)
ax1=fig.add_subplot(gs[0,0])
ax2=fig.add_subplot(gs[0,1])

for i in range(100):
    Clf.fit(X,y,itr=10)
    ax1.clear()
    Clf.Bplot(ax1,hardbound=False)
    ax2.clear()
    Clf.LCurvePlot(ax2)
    fig.canvas.draw()
    plt.pause(0.001)


