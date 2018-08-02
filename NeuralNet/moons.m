function [X, Y, s, d, angle] = moons(N, s, d, angle)
%Sample a dataset from two "moon" distributions 
%   [X, Y, s, d, angle] = moons(N, s, d, angle)
%    INPUT 
%	N     1x2 vector that fix the numberof samples from each class
%	s     standard deviation of the gaussian noise. Default is 0.1
%	d     translation vector between the two classes. With d = 0
%	      the classes are placed on a circle. Default is random.
%	angle rotation angle of the moons. Default is random.
%    OUTPUT
%	X data matrix with a sample for each row 
%   	Y vector with the labels
%
%   EXAMPLE:
%       [X, Y] = moons([10, 10]);

if (nargin <= 3)
    angle = rand(1) * pi;    
end
if (nargin <= 2)
    d = -(rand(1, 2) * 0.6) + [-0.2,-0.2];
    
end
if (nargin <= 1)
    s = 0.1;
end

d = ([cos(-angle)  -sin(-angle); sin(-angle) cos(-angle) ]*d')';

oneDSampling =  pi + rand(1,N(1))*1.3*pi + angle;
X = [sin(oneDSampling') cos(oneDSampling')] + randn(N(1),2)*s;

oneDSampling =  rand(1,N(2))*1.3*pi + angle;

X = [X; [sin(oneDSampling') cos(oneDSampling')] + randn(N(2),2)*s + repmat(d, N(2), 1)];

Y = ones(sum(N), 1);
Y(1:N(1))  = 0;
