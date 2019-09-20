
# Decision Trees
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/trees.png" width="1000"/>
</p>
[source code](https://github.com/Nikeshbajaj/spkit/blob/master/examples/trees_example.py) [jupyter-notebook](https://nbviewer.jupyter.org/github/Nikeshbajaj/spkit/blob/master/notebooks/2.3.1_Trees_Classification_Example.ipynb)

## Classification and Regression Tree
## Installation (Now in spkit library)

```pip install spkit```

```
import numpy as np
import matplotlib.pyplot as plt

from spkit.ml import ClassificationTree, RegressionTree

#for dataset
from spkit.data import dataGen as ds

```



#### Requirement: All you need for this is Numpy and matplotlib** (Of course Python >=3.0)

# See the Examples in [Jupyter-Notebook](https://github.com/Nikeshbajaj/Machine_Learning_From_Scratch/blob/master/Trees/Example-Classification_and_Regression_V2.0.ipynb) for more details

<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/tree_sinusoidal.png" width="800"/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Trees/img/a123_nik.gif" width="600"/>
</p>




### import

```
import numpy as np
import matplotlib.pyplot as plt

# Download trees.py and keep in current directory or give a path (if you know how to)
from trees import ClassificationTree, RegressionTree

# For examples
from sklearn import datasets
from sklearn.model_selection import train_test_split
```

### Iris Data
```
data = datasets.load_iris()
X = data.data
y = data.target

feature_names = data.feature_names #Optional
Xt,Xs, yt, ys = train_test_split(X,y,test_size=0.3)
```
 
 ### Initiate the classifier and train it
```
clf = ClassificationTree()

# verbose 0 for no progress, 1 for short and 2 for detailed.
# feature_names is you know, else leave it or set it to None

clf.fit(Xt,yt,verbose=2,feature_names=feature_names)  
```

### Plot the decision tree
```
# Plot Tree that has been learned
plt.figure(figsize=(15,8))
clf.plotTree(show=True)
```


### Visualizing the tree building while training

#### Iris Data (Classification)
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Trees/img/2.gif" width="400"/>
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Trees/img/4.gif" width="400"/>
</p>

#### Breast cancer Data (Classification)
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Trees/img/5.gif" width="400"/>
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Trees/img/6.gif" width="400"/>
</p>

#### Bostan House price Data (Regression)
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Trees/img/7.2.gif" width="700"/>
</p>

### Visualization of decision tree after fitting a model

**Iris data: Decesion Tree** (Option to show colored branch: Blue for True and Red for False)
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master//Trees/img/tree1_Iris.png" width="600"/>
</p> 

**Cancer data: Decesion Tree** (Or just show all branches as blue with direction to indicate True and False branch)
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master//Trees/img/tree4_Cancer.png" width="600"/>
</p> 

**Boston data: Decesion Tree**
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master//Trees/img/tree5_Boston.png" width="600"/>
</p> 


### Visualizing the progress of tree building while training

**Tree building for Cancer Data (Classification)**

***Detailed view***
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Trees/img/verbose2_1.gif" width="400"/>
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Trees/img/verbose2_2.gif" width="400"/>
</p>

***Short view***
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Trees/img/verbose1.gif" width="400"/>
</p>
