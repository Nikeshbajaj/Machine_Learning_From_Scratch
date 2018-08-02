clc
clear all

X =[[randn(100,1)-2,randn(100,1)-2]; [randn(100,1)-2,randn(100,1)+2];...
[randn(100,1)+2,randn(100,1)-2];[randn(100,1)+2,randn(100,1)+2]];

y =[zeros(100,1); 1*ones(100,1); 0*ones(100,1); 0*ones(100,1)];

W= NeuralNet(X,y,[],500,0.1,1);

DBound(X,y,W);