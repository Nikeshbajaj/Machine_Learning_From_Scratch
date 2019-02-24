# Regularization for Machine Learning
### These contents were taugh in summer school [**RegML 2016**](http://lcsl.mit.edu/courses/regml/regml2016/) by [Lorenzo Rosasco](http://web.mit.edu/lrosasco/www/) and this GUI in python was submitted as part of final exam.

#### All the coded and tested functions are in [RegML.py](https://github.com/Nikeshbajaj/Regularization_for_Machine_Learning/blob/master/RegML.py) and GUIs code structure is in [RegML_GUIv2.1.py](https://github.com/Nikeshbajaj/Regularization_for_Machine_Learning/blob/master/RegML_GUIv2.1.py)

#### [Page](https://nikeshbajaj.github.io/Regularization_for_Machine_Learning/)



#### Methods
* Regularized Least Squares -RLS [Referance](https://en.wikipedia.org/wiki/Regularized_least_squares)
* Nu-Method [Referance]()
* Iterative Landweber Method [Referance](https://en.wikipedia.org/wiki/Landweber_iteration)
* Singular Value Decomposition [Reference](https://en.wikipedia.org/wiki/Singular-value_decomposition)
* Trunctated SVD [Referance 1](http://arxiv.org/pdf/0909.4061) [Referance 2](http://langvillea.people.cofc.edu/DISSECTION-LAB/Emmie%27sLSI-SVDModule/p5module.html)
* Spectral cut-off

#### Kernal Learning 
(Linear, Polynomial, Gaussian)
* Linear ![equation1](http://latex.codecogs.com/gif.latex?%5Clarge%20K%28X%2CY%29%20%3D%20X%5ETY)
* Polynomial ![equation2](http://latex.codecogs.com/gif.latex?%5Clarge%20K%28X%2CY%29%20%3D%20%28X%5ET%20Y%20+%201%29%5Ep)
* Gaussian (RBF) ![equation3](http://latex.codecogs.com/gif.latex?%5Clarge%20K%28X%2CY%29%20%3D%20exp%5E%7B-%5Cleft%20%5C%7C%20X-Y%20%5Cright%20%5C%7C%5E2%20/%202%5Csigma%20%5E2%7D)

**K-Fold Cross Validation**

## GUI
<p align="center">
  <img src="https://raw.githubusercontent.com/Nikeshbajaj/Regularization_for_Machine_Learning/master/GUI_Win_Lin.jpg" width="800"/>
</p>

# Regularization for Machine Learning
---
## Files
1. RegML.py
2. RegML_GUIv2.1.py
3. Getting_Started_Demo.ipynb

## Requirments 
### Following libraries are required to use all the functions in RegML library
1. Python(=2.7)     
2. Numpy(>=1.10.4)     [Numpy](https://pypi.python.org/pypi/numpy) 
3. Matplotlib(>=0.98)  [Matplotlib](https://github.com/matplotlib/matplotlib) 
4. Scipy(>=0.12)       Optional -(If you need to import .mat data files)  [Scipy](https://www.scipy.org/install.html) 

## Tested with following version
GUI is tested on followwing version of libraries
* Python     2.7     **(this GUI doesn't work with python 3, need to change a lot to work with python 3)**
* Numpy      1.10.4
* Matplotlib 1.15.1
* Scipy      0.17.0

## Getting starting with GUI

### Windows------------------------
After lauching python, go to directory containing RegML.py and RegML_GUIv2.1.py files and run following command on
python shell
```
>> run RegML_GUIv2.1.py
```
If you are using Spyder or ipython qt, browes to directory, open RegML_GUIv2.1.py file and run it

### Ubuntu/Linux-------------------

Open terminal, cd to directory contaning all the files and execute following command
```
$ python RegML_GUIv2.1.py
```
if you have both python 2 and python 3

```
$ python2 RegML_GUIv2.1.py
```

If you are using Spyder or ipython qt, browes to directory, open RegML_GUIv2.1.py file and run it


## Getting Started with DEMO
Getting_Started_Demo is a IPython -Notebook, which can be open in Ipython-Notebook or Jupyter

# [**Notebook**](https://github.com/Nikeshbajaj/Regularization_for_Machine_Learning/blob/master/Getting_Started_Demo.ipynb)


# [**RegML Library**](https://github.com/Nikeshbajaj/Regularization_for_Machine_Learning/blob/master/RegML.py)

______________________

### Nikesh Bajaj

n.bajaj@qmul.ac.uk

nikesh.bajaj@elios.unige.it

[http://nikeshbajaj.in](http://nikeshbajaj.in)
