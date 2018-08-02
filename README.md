# Machine Learning From Scratch (without using any ML libraries)
## With good visualizations

## Table of contents
- [Logistic Regression](#Logistic-Regression)
- [Support Vector Machine](#SVM) (yet to implement)
- [Decision Trees](#DT)(yet to implement)
- [Deep Neural Network -DeepLearning](#Deep Neural Network)
- [Neural Network with matlab/octave](#contributing)

## 1. Logistic Regression (Python)
### Code ans examples are [here](https://github.com/Nikeshbajaj/MachineLearningFromScratch/tree/master/LogisticRegression)

```
from LogisticRegression import LR # given code
clf = LR(X,y,alpha=0.003,polyfit=True,degree=5,lambd=2)
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
    
clf.predict(X)
W,b =clf.getWeight()

```

<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/LogisticRegression/img/example5.gif" width="600"/>
  
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/LogisticRegression/img/example1.gif" width="300"/>
  <img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/LogisticRegression/img/example2.gif" width="300"/>
</p>


## 2. Deep Neural Network - Deeplearning(python)
### Code and examples are [here](https://github.com/Nikeshbajaj/DeepLearning_from_scratch)
#### Full detail of implementation and use of code is describe [here](https://github.com/Nikeshbajaj/DeepLearning_from_scratch)
 
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/DeepLearning_from_scratch/master/figures/deepnet.gif" width="300"/>
<img src="https://raw.githubusercontent.com/Nikeshbajaj/DeepLearning_from_scratch/master/figures/11.png" width="300"/>
</p>

## 3. Neural Network (simple structure) with any number of layers (Matlab/Octave) 
### Code and examples [here](https://github.com/Nikeshbajaj/MachineLearningFromScratch/tree/master/NeuralNet)

Network can be created and trained as for example
```
W= NeuralNet(X,y,HL,Iterations,alpha,verbose);

% For 2 hidden layers with 5 and 3 neurons, 500 iteration and 0.1 alpha(learning rate)
% input and output layers are chosen according to data X,y provided

W= NeuralNet(X,y,[5,3],500,0.1,1); 

% for 10 hidden layers
W= NeuralNet(X,y,[15,10,10,10,5,5,5,3],100,0.1,1);

returns weights W of each layer

```

<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/NeuralNet/Linear.bmp" width="300"/>
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/NeuralNet/NonLinear1.bmp" width="300"/>
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/NeuralNet/NonLinear3.bmp" width="300"/>
</p>



