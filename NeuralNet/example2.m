clc
clear all

[X, y] = moons([500,500],0.1,[0.5,0.5])

W= NeuralNet(X,y,[5,3],500,0.1,1);

DBound(X,y,W);