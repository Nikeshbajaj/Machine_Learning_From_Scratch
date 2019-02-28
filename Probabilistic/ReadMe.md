# Naive Bayes
### Probabilistic model
Classifier based on Bayes rule:
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Probabilistic/img/Bayes_rule.png" width="400"/>
</p>

### Example with jupyter notebook [here](https://github.com/Nikeshbajaj/Machine_Learning_From_Scratch/blob/master/Probabilistic/NaiveBayes_example.ipynb)

Notebook include example of Iris data, Breast Cancer and Digit classification (MNIST)

```
import numpy as np
import matplotlib.pyplot as plt

# For dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Library provided
from probabilistic import NaiveBayes

data = datasets.load_iris()
X = data.data
y = data.target

Xt,Xs,yt,ys = train_test_split(X,y,test_size=0.3)

print(Xt.shape,yt.shape,Xs.shape,ys.shape)

# Fitting model (estimating the parameters)
clf = NaiveBayes()
clf.fit(Xt,yt)

# Prediction
ytp = clf.predict(Xt)
ysp = clf.predict(Xs)

print('Training Accuracy : ',np.mean(ytp==yt))
print('Testing  Accuracy : ',np.mean(ysp==ys))

print(clf.parameters)

# Visualization
fig = plt.figure(figsize=(12,10))
clf.VizPx()
```
## Iris Data

<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Probabilistic/img/FeatureDist.png" width="600"/>
</p>

## Breast Cancer Data

(first 16 features)

<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Probabilistic/img/FeatureDist1.png" width="600"/>
</p>
