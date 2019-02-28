#  Neural Network (Fully Connected) with any number of layers (Matlab/Octave) 
### Code and examples [here](https://github.com/Nikeshbajaj/MachineLearningFromScratch/tree/master/NeuralNet)

Network can be created and trained as for example
```
W= NeuralNet(X,y,HL,Iterations,alpha,verbose);

% For 2 hidden layers with 5 and 3 neurons, 500 iteration and 0.1 alpha(learning rate)
% input and output layers are chosen according to data X,y provided

W= NeuralNet(X,y,[5,3],500,0.1,1); 

% for 8 hidden layers
W= NeuralNet(X,y,[15,10,10,10,5,5,5,3],100,0.1,1);

returns weights W of each layer

```

<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/NeuralNet/Linear.bmp" width="300"/>
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/NeuralNet/NonLinear1.bmp" width="300"/>
<img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/NeuralNet/NonLinear3.bmp" width="300"/>
</p>
