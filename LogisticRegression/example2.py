'''
Machine Learning from scrach
Example 2: Logistic Regression

@Author _ Nikesh Bajaj
PhD Student at Queen Mary University of London &
University of Genova
Conact _ http://nikeshbajaj.in 
n[dot]bajaj@qmul.ac.uk
bajaj[dot]nikkey@gmail.com
'''
print('running..')
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import DataSet as ds
from LogisticRegression import LR


plt.close('all')

dtype = ['MOONS','GAUSSIANS','LINEAR','SINUSOIDAL','SPIRAL']

X, y,_ = ds.create_dataset(200, dtype[3],0.05,varargin = 'PRESET')


print(X.shape, y.shape)

means = np.mean(X,1).reshape(X.shape[0],-1)
stds  = np.std(X,1).reshape(X.shape[0],-1)
X = (X-means)/stds


Clf = LR(X,y,alpha=0.003,polyfit=True,degree=5,lambd=2)

plt.ion()
delay=0.01
fig=plt.figure(figsize=(10,7))
gs=GridSpec(3,2)
ax1=fig.add_subplot(gs[0:2,0])
ax2=fig.add_subplot(gs[0:2,1])
ax3=fig.add_subplot(gs[2,:])

for i in range(100):
    Clf.fit(X,y,itr=10)
    ax1.clear()
    Clf.Bplot(ax1,hardbound=True)
    ax2.clear()
    Clf.LCurvePlot(ax2)
    ax3.clear()
    Clf.Wplot(ax3)
    fig.canvas.draw()
    plt.pause(0.001)
    #time.sleep(0.001)