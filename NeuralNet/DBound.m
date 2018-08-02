function S = DBound(X,y,W)
if size(X,2)~=2
    error('Cannot plot boundries for more than 2-dimentional input')
end

mn = min([min(X(:,1)) ,min(X(:,2))]) -1;
mx = max([max(X(:,1)) ,max(X(:,2))]) +1;

x1 = linspace(mn,mx,101);
x2 = linspace(mn,mx,101);

[xX, yY] = meshgrid(x1,x2);

XX = [xX(:),yY(:)];

%XX = [ones(size(XX,1),1), XX];

if iscell(W)
    L = size(W,2)+1;
    [cl,~] = size(W{end});
    Yi = XX';
    [rii,cii] =size(XX);
    for j =2:L
        z = W{j-1}*Yi;
        if j==L
            %disp('j==2....')
            Yi = sigmoid(z);
            %size(Yi)
        else
            %disp('j/=2....')
            Yi = [ones(1,rii);sigmoid(z)];
        end
    end
    Yi = Yi';
    if cl == 1
        yi = Yi>0.5;
        yt = [Yi,1-Yi];
    else
        [ti,yi] = max(Yi,[],2);
        yt =Yi;
    end
    
    
else
    %disp('elseeeeeeeeeeeeee')
    yt = sigmoid1(XX*W);
    if(size(W,2)==1)
        yi = yt>0.5;
        yt = [yt, 1-yt];
    else
        [~,yi] = max(yt,[],2);
    end
end

%disp('xxxxyyyyyyy')
%size(yt)

%disp('xxxdsfdsfx')
%size(Yi)

yprob = yt./sum(yt,2);
%size(yprob)

[ri,ci] = size(yprob);
sz = sqrt(ri);

%figure(1)
cls = size(yprob,2);
csl = ceil(cls/2);
S = zeros(sz,sz);
for i = 1:cls
    %subplot(2,csl,i)
    I = reshape(yprob(:,i),[sz,sz]);
    S = max(S,I);
    %surf(I*255)
end

figure(2)
subplot(1,2,1)
surfc(S)
title('Surface Plot of Boundries')
xlabel('x1')
ylabel('x2')
zlabel('Probability')

figure(2)
subplot(1,2,2)
c = unique(y);
color = ['r','b','g','m','c','k','c'];
for i = 1:length(c)
    plot(X(y==c(i),1),X(y==c(i),2),['.',color(i)],'MarkerSize',8)
    hold on
end

c = unique(yi);
for i = 1:length(c)
    %plot(XX(yi==c(i),2),XX(yi==c(i),3),['.',color(i)],'MarkerSize',3)
    plot(XX(yi==c(i),1),XX(yi==c(i),2),['.',color(i)],'MarkerSize',3)
end
hold off
xlim([mn,mx])
ylim([mn,mx])
xlabel('x1')
ylabel('x2')
title('Decision broundries')
end

function y = sigmoid(x)
[r,c] =size(x);
y = zeros(r,c)-1;
for i =1:r
    y(i,:) = 1./(1 + exp(-x(i,:)));
end
end
