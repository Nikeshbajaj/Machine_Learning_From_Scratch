function W = NeuralNet(X,y,HL,itrR,alpha,verbose)

  %[n, f] = size(X);
  %---Add extra feature of ones to avoid bais term
  %if(sum(X(:,1)==1)~=n)
  %  X = [ones(n,1),X];
  %end 
  
  [n, f] = size(X);
  cl = unique(y);
  nC = length(cl);
  
  fprintf('Number of datapoints %d and features %d \n',n,f)
  fprintf('Number of classes %d \n',nC)
  fprintf('Class Labels %d \n',cl)
   
  iL  = f;
  oL  = nC;
  if oL==2, oL =1; end
  
  disp('Network: ')
  Net = [iL, HL, oL]
  L = length(Net);
  fprintf('Size of network %d',L);
  
  % --- Intializing weights W and dW (if want to update after batch)-----------
  for j = 1:L-1
    if j==1
      W{j} = randn(Net(j+1),Net(j))*sqrt(2/Net(j));
      %dW{j}= zeros(Net(j+1),Net(j));
    else
      W{j} = randn(Net(j+1),Net(j)+1);
      %dW{j}= zeros(Net(j+1),Net(j)+1);
    end
  end 
 
  %-----One Hot vector for more than 2 classes---------
  if oL>2
    for i =1:oL
      t(y==i-1,i)=1; 
    end
  else 
    t=y;
  end
  
  
  Acc =[];
  n1 = n/4;  % Batch Size set of one forth of all the data points-----
  for itr =1:itrR
   pix = randperm(n);  % in case randperm doesn't work use "randPerm" as defined below
   for i = 1:n1
     k = pix(i);
     xi = X(k,:)';
     %ai={};
     ai{1} = xi;

     % Forward Propogation---->>
     for j =2:L
       z = W{j-1}*ai{j-1};
       if j==L
        ai{j} = sigmoid(z);
       else
        ai{j} = [1;sigmoid(z)];
       end
     end
     
     % Grad Computation -- Backward  <<------
     del{L} = -(t(k,:)' - ai{L}).*ai{L}.*(1-ai{L});
     for j = L-1:-1:1
     dd = del{j+1};
     if j+1<L; dd(1) =[]; end
     ww = W{j};
     del{j} = ww'*dd.*ai{j}.*(1-ai{j});
     end
  
     % Weight Change Computation ------
     for  j =L-1:-1:1
       dd = del{j+1};
       if j+1<L; dd(1) =[]; end
       DW{j} = dd*(ai{j})';
       %dW{j}= dW{j}+DW{j};
     end
     
     % Wieght Updation -------  Weight updation can be done after every
     % batch or full batch, we are doing it here after every iteration
     % which is like stochastic gradient decent approach
     
     for j =1:L-1
        W{j} = W{j} - alpha*DW{j};
     end
   end
   
    %for j =1:L-1
        %W{j} = W{j} - alpha*dW{j}/n1;
    %end
   
   %disp('Error............')
    Yi = X';
    [rii,cii] =size(X);
    
     for j =2:L
       z = W{j-1}*Yi;
       if j==L
          Yi = sigmoid(z);
       else
          Yi = [ones(1,rii);sigmoid(z)];
       end
     end
   Yi = Yi';
   %[t,Yi]; 
   if(size(t,2)==1)
       yt = Yi>0.5;
       tt = yt==t;
       ti = Yi;
   else
      [ti,yt] = max(Yi,[],2);
       yt = yt-1;
       tt = yt==y;
   end
   Acc =[Acc sum(tt(:))/length(tt(:))];
   if(verbose==true)
     fprintf('Epoc : %d  Accuracy : %f \n',itr,Acc(end))
   end
   
  end
  fprintf('Epoc : %d  Accuracy : %f \n',itr,Acc(end))
  plot(Acc*100,'r','MarkerSize',10)
  grid('on')
  axis([1,length(Acc),0,100])
  xlabel('Iteration')
  ylabel('Accuracy')
  title('Accuracy over iterations')
  %[y,yt,ti]
end


function y = sigmoid(x)
       [r,c] =size(x);
       y = zeros(r,c)-1;
       for i =1:r
       y(i,:) = 1./(1 + exp(-x(i,:)));
       end
end


function p = randPerm(x)
    p=[];
    while(~isempty(x))
    l = length(x);
    ri = randi(1,l);
    p =[p x(ri)];
    x(ri)=[];
    end
end