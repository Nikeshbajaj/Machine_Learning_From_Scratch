import numpy as np
import matplotlib.pyplot as plt
import random
import tkMessageBox


##------------List of Functions--------------- return----------------------
'''
EuclideanDistance(a,b,df=0):                   return d                                                            ### Tested 
autosigma(X, K):                               return s                                                            ### Tested 
kernel(knl, kpar, X1, X2):                     return k                                                            ### Tested 
SVD(X):                                        return U,S,V                                                        ### Tested 
cutoff(K, t_range, y):                         return alpha                                                        ### Tested 
kcv(knl, kpar, filt, t_range, X, y, k, task, split_type):          return t_kcv_idx, avg_err_kcv                   ### Tested 
land(K, t_max, y, tau, all_path):              return alpha                                                        ### Tested
learn_error(y_learnt, y_test, learn_task):     return lrn_error                                                    ### Tested
learn(knl, kpar, filt, t_range, X, y, task):   return alpha, err                                                   ### Tested
nu(K, t_max, y, all_path):                     return alpha                                                        ### Tested 
patt_rec(knl, kpar, alpha, x_train, x_test, y_test, learn_task):   return y_lrnt, test_err                         ### to be Tested 
rls(K, t_range, y):                            return alpha                                                        ### Tested 
splitting(y, k, type = 'seq'):                 return sets                                                         ### Tested 
tsvd(K, t_range, y):                           return alpha                                                        ### Tested

#-----------Dataset Generators------------------------
spiral(N, s = 0.5, wrappings = 'random', m = 'random')                                                             ### Tested
sinusoidal(N, s = 0.1)                                                                                             ### Tested
moons(N, s =0.1, d='random', angle = 'random')                                                                     ### Tested
gaussian(N, ndist = 3, means ='random', sigmas='random')                                                           ### Tested
linear_data(N, m ='random', b ='random', s =0.1)                                                                   ### Tested

#------------Reading .mat file-------------------------

loadMatFile(filePath)                                                                                              ### Tested

#-------Plotting Funtions--------------------------------
plot_dataset(X, Y)                                                                                                 ### Tested
plot_estimator(alpha, xTrain, yTrain, xTest, yTest, ker, par)                                                      ### to be Tested

------------------------------------------------------------------------------------------------------------------------------------
For any bugs or fixs, please let me know
Nikesh Bajaj
nikesh.bajaj@elios.unige.it
n.bajaj@qmul.ac.uk

'''
##--------------------------------------------------------------------------------------------------------------------------------
# For all the examples in description considered that numpy library is imported as np
#--------------------------
#   import numpy as np
#--------------------------
# Remarks for Developers - Types of data structure
#  X, y, K --> Numpy array
#  t_range --> List or numpy array or scalar
#  alpha   --> List of arrays

def EuclideanDistance(a,b,df=0):
    '''
    % EUCLIDEAN - computes Euclidean distance matrix
    %
    % E = EuclideanDistance(A,B)
    %
      Input
    %    a - MxD numpy array (matrix)
         b - NxD numpy array (matrix)
    %    df = 0  (default), do not force
              1  force diagonals to be zero; 
    % 
    % Returns:
    %    E - (MxN) Euclidean distances between vectors in A and B
    %
    %
    % Description : 
    %    This fully vectorized (VERY FAST!) m-file computes the 
    %    Euclidean distance between two vectors by:
    %    A = a.T, B = b.T
    %                 ||A-B|| = sqrt ( ||A||^2 + ||B||^2 - 2*A.B )
    %
    % Example :
         
    %    A = np.random.rand(100,400)
         B = np.random.rand(200,400);
    %    d = EuclideanDistance(A,B);
         d = EuclideanDistance(A,B,df=1);
    %     

    % Author   : Roland Bunschoten
    %            University of Amsterdam
    %            Intelligent Autonomous Systems (IAS) group
    %            Kruislaan 403  1098 SJ Amsterdam
    %            tel.(+31)20-5257524
    %            bunschot@wins.uva.nl
    % Last Rev : Wed Oct 20 08:58:08 MET DST 1999
    % Tested   : PC Matlab v5.2 and Solaris Matlab v5.3

    % Copyright notice: You are free to modify, extend and distribute 
    %    this code granted that the author of the original code is 
    %    mentioned as the original author of the code.

    % Fixed by JBT (3/18/00) to work for 1-dimensional vectors
    % and to warn for imaginary numbers.  Also ensures that 
    % output is all real, and allows the option of forcing diagonals to
    % be zero.
    '''
    a = a.T #a=a';   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< to make   A - (DxM) matrix  
    b = b.T #b=b';                                                        B - (DxN) matrix
    '''
    if (size(a,1) == 1)<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Add this
      a = [a; zeros(1,size(a,2))]; 
      b = [b; zeros(1,size(b,2))]; 
    end
    '''
    
    if a.shape[0] != b.shape[0]:
        tkMessageBox.showerror("Error","A and B should be of same dimensionality")
        raise ValueError("A and B should be of same dimensionality",'Tips and tricks')

    if (np.sum(np.iscomplex(a)) + np.sum(np.iscomplex(b))) > 0:
        tkMessageBox.showinfo("Warning","running L2Distance.m with imaginary numbers.  Results may be off")

    #aa = np.multiply(a,a).sum(axis=0)
    #aa = aa.reshape((1,aa.shape[0]))    
    aa = np.array([np.multiply(a,a).sum(axis=0)])   #replacement of above two lines
    
    #bb = np.multiply(b,b).sum(axis=0)
    #bb = bb.reshape((1,bb.shape[0]))
    bb = np.array([np.multiply(b,b).sum(axis=0)])   #replacement of above two lines
    
    ab = np.matmul(a.T, b)

    d = np.sqrt(np.tile(aa.T,(1,bb.shape[1])) + np.tile(bb,(aa.shape[1],1)) - 2*ab)

    #make sure result is all real
    d = np.real(d); 

    # force 0 on the diagonal? 
    if (df==1):
        d = np.multiply(d,1-np.eye(d.shape[0],d.shape[1]))
    return d

def autosigma(X, K):
    '''
    %AUTOSIGMA Compute the average K nearest neighbor distance of n p-dimensional points.
    %   S = AUTOSIGMA(X, K) calculate the average K nearest neighbor
    %   distance of n p-dimensional given a data matrix 'X[n,p]' and a number 
    %   'K' of nearest neighbors returned by the K-NN
    %   
    %   Example:
    %        s = autosigma(X, 5);
    %
    % See also KERNEL
    '''    
    E = EuclideanDistance(X,X,1);
    E = np.sort(E,axis=0);
    s = np.mean(E[K+1,:]);
    
    return s

def kernel(knl, kpar, X1, X2):
    '''
    %KERNEL Calculates a kernel matrix.
    %   K = KERNEL(KNL, KPAR, X1, X2) calculates the nxN kernel matrix given
    %   two matrix X1[n,d], X2[N,d] with kernel type specified by 'knl':
    %       'lin'   - linear kernel, 'kpar' is not considered
    %       'pol'   - polinomial kernel, where 'kpar' is the polinomial degree
    %       'gauss' - gaussian kernel, where 'kpar' is the gaussian sigma
    %
    %   Example:
    %       X1 = np.random.randn(n, d)
    %       X2 = np.random.randn(N, d)
    %       K = kernel('lin', [], X1, X2)     OR  K = kernel(knl = 'lin',   kpar = [],  X1 = X1, X2 = X2)
    %       K = kernel('gauss', 2.0, X1, X2)  OR  K = kernel(knl = 'gauss', kpar = 2.0, X1 = X1, X2 = X2)
    %
    % See also LEARN
    '''
    #print knl, kpar, X1.shape, X2.shape
    
    N = X1.shape[0]
    n = X2.shape[0]
    
    if knl.lower()=='lin':
        #print 'Linear Kernal'
        
        k = np.dot(X1,X2.T)
        
    elif knl.lower()=='pol':
        #print 'Polynomial Kernal'
        deg = kpar
        if int(deg)!=deg or deg < 1:
            #error('Polynomial kernel degree should be an integer greater or equal than 1','Tips and tricks')
            tkMessageBox.showerror("Tips and tricks","Polynomial kernel degree should be an integer greater or equal than 1")
            raise ValueError('Polynomial kernel degree should be an integer greater or equal than 1','Tips and tricks')
        else:
            k = np.power(np.dot(X1,X2.T) + 1 , deg)
    
    elif knl.lower()=='gauss':
        #print 'Gaussian Kernal'
        sigma = kpar
        if sigma<=0:
            #error('Gaussian kernel sigma should be greater than 0','Tips and tricks')
            tkMessageBox.showerror("Tips and tricks","Gaussian kernel sigma should be greater than 0")
            raise ValueError('Gaussian kernel sigma should be greater than 0','Tips and tricks')
        
        var = 2*sigma*sigma
        sqx = np.multiply(X1,X1).sum(axis=1).reshape((X1.shape[0],1))
        sqy = np.multiply(X2,X2).sum(axis=1).reshape((X2.shape[0],1))
        k = np.dot(sqx,np.ones([1, n])) + np.dot(np.ones([N,1]),sqy.T) - 2*np.dot(X1,X2.T)
        k=np.exp(-(k/var))
    else:
        print 'Unknown Kernal'
        tkMessageBox.showerror("Tips and tricks",'Unknown kernel! Choose the appropriate one!')
        raise ValueError('Unknown kernel! Choose the appropriate one!','Tips and tricks')
      
    return k

def SVD(X):
    '''
    Singular Value Decomposition, 
    returns full matrixes without truncating zeros from S matrix
    
    Input:
       X - MxN
    return 
       U - MxM
       S - MxN
       V - NxN
    '''
    U,s,V = np.linalg.svd(X,full_matrices=True)
    S = np.zeros(X.shape)
    c = s.shape[0]
    S[:c,:c] = np.diag(s)
    return U,S,V.T

def cutoff(K, t_range, y):
    '''
    %CUTOFF Calculates the coefficient vector using cutoff method.
    %   [ALPHA] = CUTOFF(K, T_RANGE, Y) calculates the spectral cut-off 
    %   solution of the problem 'K*ALPHA = Y' given a kernel matrix 'K[n,n]' a 
    %   range of regularization parameters 'T_RANGE' and a label/output 
    %   vector 'Y'.
    %
    %   The function works even if 'T_RANGE' is a single value
    %
    %   Example:
    %       K = kernel('lin', [], X, X);
    %       alpha = cutoff(K, np.logspace(-3, 3, 7), y);
    %       alpha = cutoff(K, 0.1, y);
    %
    % See also RLS, NU, TSVD, LAND
    '''
    n = np.array(y).shape[0]
    alpha =[]
    U,S,V = SVD(K)
    ds = np.diag(S)
    
    if np.isscalar(t_range):
        t_range = [t_range]
    
    for i in range(len(list(t_range))):
        t = t_range[i]
        mask = ds>t*n
        index = np.sum(mask,axis=0)+1
        
        inv_ds = np.vstack([np.array([1.0/ds[:index-1]]).T, 1.0/(t*n)*np.ones([n-index+1, 1])])
        
        TinvS = np.diag(inv_ds[:,0])
        
        # TinvS = np.diag(list(1.0/ds[:index-1]) +list(1.0/(t*n)*np.ones([1,n-index+1])[0]))  #Replacement of above 2 lines
        
        TK = np.dot(V,np.dot(TinvS,U.T))
        
        alpha.append(np.dot(TK,y))
        
    return alpha
     
def kcv(knl, kpar, filt, t_range, X, y, k, task, split_type):
    '''
    %KCV Perform K-Fold Cross Validation.
    %   [T_KCV_IDX, ERR_KCV] = KCV(KNL, KPAR, FILT, T_RANGE, X, Y, K, TASK, SPLIT_TYPE) 
    %   performs k-fold cross validation to calculate the index of the 
    %   regularization parameter 'T_KCV_IDX' within a range 'T_RANGE' 
    %   which minimizes the average cross validation error 'AVG_ERR_KCV' given 
    %   a kernel type 'KNL' and (if needed) a kernel parameter 'KPAR', a filter
    %   type 'FILT' and a dataset composed by the input matrix 'X[n,d]' and the output vector 
    %   'Y[n,1]'.
    %
    %   The allowed values for 'KNL' and 'KPAR' are described in the
    %   documentation given with the 'KERNEL' function. Moreover, it is possible to
    %   specify a custom kernel with 'KNL='cust'' and 'KPAR[n,n]' matrix.
    %
    %   The allowed values for 'FILT' are:
    %       'rls'   - regularized least squares
    %       'land'  - iterative Landweber
    %       'tsvd'  - truncated SVD
    %       'nu'    - nu-method
    %       'cutoff'- spectral cut-off
    %   
    %   The parameter 'T_RANGE' may be a range of values or a single value.
    %   In case 'FILT' equals 'land' or 'nu', 'T_RANGE' *MUST BE* a single
    %   integer value, because its value is interpreted as 'T_MAX' (see also 
    %   LAND and NU documentation). Note that in case of 'land' the algorithm
    %   step size 'tau' will be automatically calculated (and printed).
    %
    %   According to the parameter 'TASK':
    %       'class' - classification
    %       'regr'  - regression
    %   the function minimizes the classification or regression error.
    %
    %   The last parameter 'SLIT_TYPE' must be:
    %       'seq' - sequential split (as default)
    %       'rand' - random split
    %   as indicated in the 'SPLITTING' function.
    %
    %   Example:
    %       t_kcv_idx, avg_err_kcv = kcv('lin', [], 'rls', np.logspace(-3, 3, 7), X, y, 5, 'class', 'seq')
    %       t_kcv_idx, avg_err_kcv = kcv('gauss', 2.0, 'land', 100, X, y, 5, 'regr', 'rand')
    %       
    
    OR---------
    t_kcv_idx, avg_err_kcv = kcv(knl = 'lin', kpar = [], filt = 'rls', t_range =[3,5], X =X1, y =y1, k = 5, task ='class', split_type ='seq')
    

    % See also LEARN, KERNEL, SPLITTING, LEARN_ERROR
    '''
    k = int(np.ceil(k))
    if (k <= 1):
        tkMessageBox.showerror("Tips and tricks",'The number of splits in KCV must be at least 2')
        raise ValueError('The number of splits in KCV must be at least 2','Tips and tricks');
    
    #Split of training set:
    sets = splitting(y, k, split_type)
    
    #Starting Cross Validation
    err_kcv = [[] for _ in range(k)]
    
    
    for split in range(k):
        
        print 'split number :', split+1
        
        test_idxs = sets[split];
        train_idxs = np.setdiff1d(np.arange(y.shape[0]), test_idxs)
        
        X_train = X[train_idxs, :]
        y_train = y[train_idxs]

        X_test = X[test_idxs, :]
        y_test = y[test_idxs]
        
        # Learning
        alpha,er =  learn(knl, kpar, filt, t_range, X_train, y_train, task)
        
        ##Test error estimation
        # Error estimation over the test set, using the parameters given by the
        # pprevious task
        K_test = kernel(knl, kpar, X1=X_test, X2 = X_train)
        
        # On each split we estimate the error with each t value in the range
        
        err_kcv[split] = np.zeros(len(alpha))
        for t in range(len(alpha)):
            y_learnt = np.dot(K_test,alpha[t])
            err_kcv[split][t] = learn_error(y_learnt, y_test, task)

    # Average the error over different splits
    
    avg_err_kcv = np.median(np.vstack(err_kcv), axis=0)
    
    # Calculate minimum error w.r.t. the regularization parameter
    
    t_kcv_idx = np.argmin(avg_err_kcv)
    
    #print alpha[t_kcv_idx].shape
    #print alpha[t_kcv_idx]
    
    return t_kcv_idx, avg_err_kcv

def land(K, t_max, y, tau, all_path = False):
    '''
    %LAND Calculates the coefficient vector using Landweber method.
    %   [ALPHA] = LAND(K, T_MAX, Y, TAU) calculates the regularized least 
    %   squares  solution of the problem 'K*ALPHA = Y' given a kernel matrix 
    %   'K[n,n]' a maximum regularization parameter 'T_MAX', a
    %   label/output vector 'Y' and a step size 'TAU'.
    %
    %   [ALPHA] = LAND(K, T_MAX, Y, TAU, ALL_PATH) returns only the last 
    %   solution calculated using 'T_MAX' as regularization parameter if
    %   'ALL_PATH' is false. Otherwise return all the regularization path.
    %
    %   Example:
    %       K = kernel('lin', [], X, X);
    %       alpha = land(K, 10, y, 2);
    %       alpha = land(K, 10, y, 2, true);
    %
    % See also NU, TSVD, CUTOFF, RLS
    '''
    t_max = int(np.floor(t_max))
    if(t_max<1):
        tkMessageBox.showerror("Tips and tricks",'t_max must be an int greater than 0')
        raise ValueError('t_max must be an int greater than 0','Tips and tricks')
    if(len(y.shape)==1):
        y = np.array([y]).T
    
    n = np.array(y).shape[0]
    
    alpha = [[] for _ in range(t_max)] 
    
    alpha[0] = np.zeros([n,1])
    
    for j in range(1,t_max):
        alpha[j] = alpha[j-1] + (tau/float(n))*(y - np.dot(K,alpha[j-1]))
  
    if(not(all_path)):
        alpha = [alpha[t_max-1]]
        
    return alpha

def learn_error(y_learnt, y_test, learn_task):
    '''
    %LEARN_ERROR Compute the learning error.
    %   LRN_ERROR = LEARN_ERROR(Y_LEARNT, Y_TEST, LEARN_TASK) computes the 
    %   classification or regression error given two vectors 'Y_LEARNT' and 
    %   'T_TEST', which contain respectively the learnt and the the test 
    %   labels/values of the output set.
    %   The parameter 'LEARN_TASK' specify the kind of error:
    %       'regr' - regression error
    %       'class' - classification error
    %
    %   Example:
    %       y_learnt = Kernel * alpha;
    %       lrn_error = learn_error(y_learnt, y_test, 'class');
    %       lrn_error = learn_error(y_learnt, y_test, 'regr');
    %
    % See also LEARN
    '''
    if learn_task =='class':
        lrn_error = np.sum(np.multiply(y_learnt, y_test)<=0) / float(y_test.shape[0])
        
    elif learn_task =='regr':
        lrn_error = np.sum((y_learnt - y_test)**2) / float(y_test.shape[0])
        
    else:
        tkMessageBox.showerror("Tips and tricks",'Unknown learning task!')
        raise ValueError('Unknown learning task!','DEBUG: YOU ARE FAILING AT FAIL!!')
    return lrn_error

def learn(knl, kpar, filt, t_range, X, y, task = False):
    '''
    %LEARN Learns an estimator. 
    %   ALPHA = LEARN(KNL, KPAR, FILT, T_RANGE, X, Y) calculates a set of 
    %   estimators given a kernel type 'KNL' and (if needed) a kernel parameter 
    %   'KPAR', a filter type 'FILT', a range of regularization parameters 
    %   'T_RANGE' and a training set composed by the input matrix 'X[n,d]' and
    %   the output vector 'Y[n,1]'.
    %
    %   The allowed values for 'KNL' and 'KPAR' are described in the
    %   documentation given with the 'KERNEL' function. Moreover, it is possible to
    %   give a custom kernel with 'KNL='cust'' and 'KPAR[n,n]' matrix.
    %
    %   The allowed values for 'FILT' are:
    %       'rls'   - regularized least squares
    %       'land'  - iterative Landweber
    %       'tsvd'  - truncated SVD
    %       'nu'    - nu-method
    %       'cutoff'- spectral cut-off
    %
    %   The parameter 'T_RANGE' may be a range of values or a single value.
    %   In case of 'FILT' equal 'land' or 'nu', 'T_RANGE' *MUST BE* a single
    %   integer value, because its value is interpreted as 'T_MAX' (see also 
    %   LAND and NU documentation). Note that in case of 'land' the algorithm
    %   step size 'tau' will be automatically calculated (and printed).
    %
    %   [ALPHA, ERR] = LEARN(KNL, FILT, T_RANGE, X, Y, TASK) also returns
    %   either classification or regression errors (on the training data)
    %   according to the parameter 'TASK':
    %       'class' - classification
    %       'regr'  - regression
    %
    %   Example:
    %       alpha = learn('lin', [], 'rls', logspace (-3, 3, 7), X, y);
    %       alpha = learn('gauss', 2.0, 'land', 100, X, y);
    %       [alpha, err] = learn('lin', [] , 'tsvd', logspace(-3, 3, 7), X, y, 'regr');
    %
    %   See also KCV, KERNEL, LEARN_ERROR
    '''
    
    if ((filt=='nu' or filt=='land') and not(np.isscalar(t_range))):
        tkMessageBox.showerror("Tips and tricks",'The dimension of the t_range array MUST be 1')
        raise ValueError('The dimension of the t_range array MUST be 1','Tips and tricks')
    if not(task =='class' or task =='regr'):
        tkMessageBox.showerror("Tips and tricks",'Unknown learning task!')
        raise ValueError('Unknown learning task!','DEBUG: YOU ARE FAILING AT FAIL!!')
    # Compute Kernal k
    if knl=='cust':
        n = X.shape[0]
        if kpar.shape !=(n,n):
            tkMessageBox.showerror("Tips and tricks",'Not valid custom kernel')
            raise ValueError('Not valid custom kernel','Tips and tricks')
    K = kernel(knl,kpar,X,X)
    
    if filt.lower() =='land':
        if knl.lower()=='gauss':
            tau =2
        else:
            w,_ = np.linalg.eig(K)
            s = w[0]
            tau =2.0/s
        tau = np.real(tau)
        print 'Calculated the step size tau : ', tau
        
        #land(K, t_max, y, tau, all_path = False)
        alpha = land(K, t_range, y, tau, all_path =True)
    
    
    elif filt.lower() =='nu':
        #nu(K, t_max, y, all_path = False)
        alpha = nu(K, t_range, y, all_path =True)
    
    elif filt.lower() =='rls':        
        #rls(K, t_range, y)
        alpha = rls(K, t_range, y)
        
    elif filt.lower() =='tsvd':
        #tsvd(K, t_range, y)
        alpha = tsvd(K, t_range, y)
    
    elif filt.lower() =='cutoff':
        #cutoff(K, t_range, y)
        alpha = cutoff(K, t_range, y)
        
    else:
        tkMessageBox.showerror("Tips and tricks", 'Unknown filter. Please specify one in: nu, rls, tsvd, land, cutoff')
        raise ValueError('Unknown filter. Please specify one in: nu, rls, tsvd, land, cutoff','Tips and tricks')
    
    err = [[] for _ in range(len(alpha))]
    if task:
        for i in range(len(alpha)):
            y_lrnt = np.dot(K,alpha[i])
            err[i] = learn_error(y_lrnt, y,task)
            
    return alpha,err

def nu(K, t_max, y, all_path = False):
    '''
    %NU Calculates the coefficient vector using NU method.
    %   [ALPHA] = NU(K, T_MAX, Y) calculates the solution of the problem 
    %   'K*ALPHA = Y' using NU method given a kernel matrix 
    %   'K[n,n]', the maximum number of the iterations 'T_MAX' and a 
    %   label/output vector 'Y'.
    %
    %   [ALPHA] = NU(K, T_MAX, Y, ALL_PATH) returns only the last 
    %   solution calculated using 'T_MAX' as regularization parameter if
    %   'ALL_PATH' is false(DEFAULT). Otherwise return all the regularization 
    %   path.
    %
    %   Example:
    %       K = kernel('lin', [], X, X);
    %       alpha = nu(K, 10, y);
    %       alpha = nu(K, 10, y, true);
    %
    % See also TSVD, CUTOFF, RLS, LAND
    '''
    #print'-------------------------------------'
    #print t_max
    t_max = int(np.floor(t_max))
    #print t_max
    if(t_max<3):
        tkMessageBox.showerror("Tips and tricks",'t_max must be an int greater than 2')
        raise ValueError('t_max must be an int greater than 2','Tips and tricks')
    if(len(y.shape)==1):
        y = np.array([y]).T
    
    n = np.array(y).shape[0]
    
    alpha = [[] for _ in range(t_max)] 
    
    alpha[0] = np.zeros([n,1])
    alpha[1] = np.zeros([n,1])
    nu=1
    
    for j in range(2,t_max):
        i=float(j+1)
        u = ((i-1) * (2*i-3) * (2*i+2*nu-1)) / ((i+2*nu-1) * (2*i+4*nu-1) * (2*i+2*nu-3))
        w = 4 * ( ((2*i+2*nu-1)*(i+nu-1)) / ((i+2*nu-1)*(2*i+4*nu-1)))
        alpha[j] = alpha[j-1] + u*(alpha[j-1] - alpha[j-2]) + (w/n)*(y - np.dot(K,alpha[j-1]))
     
    if(not(all_path)):
        #print ' '
        alpha = [alpha[t_max-1]]
        
    return alpha

def patt_rec(knl, kpar, alpha, x_train, x_test, y_test, learn_task):
    '''
    %PATT_REC Calculates a test error given a training and a test dataset and an estimator
    %   [Y_LRNT, TEST_ERR] = PATT_REC(KNL, KPAR, ALPHA, X_TRAIN, X_TEST, Y_TEST)
    %   calculates the output vector Y_LRNT and the test error (regression or
    %   classification) TEST_ERR given
    %
    %      - kernel type specified by 'knl':
    %        'lin'   - linear kernel, 'kpar' is not considered
    %        'pol'   - polinomial kernel, where 'kpar' is the polinomial degree
    %        'gauss' - gaussian kernel, where 'kpar' is the gaussian sigma
    %
    %      - kernel parameter 'kpar':
    %        'deg'  - for polynomial kernel 'pol'
    %        'sigma' - for gaussian kernel 'gauss'
    %
    %      - an estimator 'alpha'
    %        training set 'x_train'
    %        test set 'x_test'
    %        known output labels/test data 'y_test'
    %
    %      - a learn_task 'learn_task'
    %        'class' - for classification
    %        'regr' - for regression
    %    
    %   Example:
    %    [y_lrnt, test_err] = patt_rec('gauss', .4, alpha,x, x_test, y_test)
    %
    % See also LEARN, KERNEL, LEARN_ERROR
    '''
    
    K_test = kernel(knl,kpar,x_test,x_train)           # Compute test kernel
    y_lrnt = np.dot(K_test , alpha)                    # Computre predicted output vector

    test_err = learn_error(y_lrnt, y_test, learn_task) # Evaluate error
    
    
    return y_lrnt, test_err

def rls(K, t_range, y):
    '''
    %   RLS Calculates the coefficient vector using Tikhonov method.
    %   [ALPHA] = RLS(K, lambdas, Y) calculates the least squares solution
    %   of the problem 'K*ALPHA = Y' given a kernel matrix 'K[n,n]' a 
    %   range of regularization parameters 'lambdas' and a label/output 
    %   vector 'Y'.
    %
    %   The function works even if 'T_RANGE' is a single value
    %
    %   Example:
    %       K = kernel('lin', [], X, X);
    %       alpha = rls(K, logspace(-3, 3, 7), y);
    %       alpha = rls(K, 0.1, y);
    %
    % See also NU, TSVD, LAND, CUTOFF
    '''
    n = np.array(y).shape[0]
    if np.isscalar(t_range):
        t_range = [t_range]
    
    alpha = [[] for _ in range(len(list(t_range)))]
    [U,S,V] = SVD(K)
    ds=np.diag(S)
                                            ## Considering t_range--> list or np.array to be fixed if t_range = 0.1
                                            #   ----------------Fixed--------------
    
        
    for i in range(len(list(t_range))):
        lamd = t_range[i]
        Inv = 1.0/ (ds + lamd*n)     #-----------------------Check and Fix------------_Fixeddd
        TikS = np.diag(Inv)   
        TikK = np.dot(V, np.dot(TikS, U.T))
        alpha[i] = np.dot(TikK, y)
    return alpha

def splitting(y, k, type = 'seq'):
    '''
    % SPLITTING Calculate cross validation splits.
    %   SETS = SPLITTING(Y, K) splits a dataset to do K-Fold Cross validation 
    %   given a labels vector 'Y', the number of splits 'K'.
    %   Returns a cell array of 'K' subsets of the indexes 1:n, with 
    %   n=length(Y). The elements 1:n are split so that in each 
    %   subset the ratio between indexes corresponding to positive elements 
    %   of array 'Y' and indexes corresponding to negative elements of 'Y' is 
    %   the about same as in 1:n. 
    %   As default, the subsets are obtained  by sequentially distributing the 
    %   elements of 1:n.
    %
    %   SETS = SPLITTING(Y, K, TYPE) allows to specify the 'TYPE' of the
    %   splitting of the chosen from
    %       'seq' - sequential split (as default)
    %       'rand' - random split
    %
    %    Example:
    %       sets = splitting(y, k);
    %       sets = splitting(y, k, 'rand');
    %
    % See also KCV
    '''
    
    if k <=0:
        tkMessageBox.showerror("Tips and tricks",'Parameter k MUST be an integer greater than 0')
        raise ValueError('Parameter k MUST be an integer greater than 0','Tips and tricks')
    if type not in ['seq','rand']:
        tkMessageBox.showerror("Tips and tricks",'type must be seq or rand','DEBUG: FOR DEVELOPERS EYES ONLY')
        raise ValueError('type must be seq or rand','DEBUG: FOR DEVELOPERS EYES ONLY')
    
    
    n= np.array(y).shape[0]
    
    if k==n:
        sets = range(n)
    
    
    else:
        sets = [[] for _ in range(k)]
        
        c1 = np.where(y >=0)[0]
        c2 = np.where(y <0)[0]

        if type=='rand':
            c1 = np.random.permutation(c1)
            c2 = np.random.permutation(c2)
        
        c1 = list(c1)
        c2 = list(c2)
        
        while(len(c1)>0):
            for s in sets:
                if len(c1)>0:
                    s.append(c1.pop(0))
                    
        while(len(c2)>0):
            for s in sets:
                if len(c2)>0:
                    s.append(c2.pop(0))
    return sets

def tsvd(K, t_range, y):
    '''
    %TSVD Calculates the coefficient vector using TSVD method.
    %   [ALPHA] = TSVD(K, T_RANGE, Y) calculates the truncated singular values
    %   solution of the problem 'K*ALPHA = Y' given a kernel matrix 'K[n,n]' a 
    %   range of regularization parameters 'T_RANGE' and a label/output 
    %   vector 'Y'.
    %
    %   The function works even if 'T_RANGE' is a single value
    %
    %   Example:
    %       K = kernel('lin', [], X, X);
    %       alpha = tsvd(K, logspace(-3, 3, 7), y);
    %       alpha = tsvd(K, 0.1, y);
    %
    % See also RLS, NU, LAND, CUTOFF
    '''
    
    
    errmsg1 = 'T-SVD: The range of t must be (0,s] in case of "Linear Space"; Otherwise for "Log Space" must be (-inf,log s],'
    errmsg2 = "with s the biggest eigenvalue of the kernel matrix ,Tips and tricks"
    
    if np.isscalar(t_range):
        t_range = [t_range]
    
    if (sum(np.array(t_range) > 1) + sum(np.array(t_range)<0)):
        tkMessageBox.showerror("Tips and tricks",errmsg1+errmsg2)
        raise ValueError(errmsg1+errmsg2)
    
    
    n = np.array(y).shape[0]
    alpha = [[] for _ in range(len(list(t_range)))]
    [U,S,V] = SVD(K)
    ds=np.diag(S)
    
    for i in range(len(list(t_range))):
        t = t_range[i]
        mask  = ds > (t*n)
        
        Inv = np.multiply(1.0/ds , mask)
        
        TikS = np.diag(Inv)   
        
        TikK = np.dot(V, np.dot(TikS, U.T))
        
        alpha[i] = np.dot(TikK, y)

    return alpha

##------------------------------Dataset Generators---------------------------------------------

def spiral(N, s = 0.5, wrappings = 'random', m = 'random'):
    '''
    %Sample a dataset from a dataset separated by a sinusoidal line
    %   X, Y, s, wrappings, m = spiral(N, s, wrappings, m)
    %    INPUT 
    %	N         1x2 vector that fix the numberof samples from each class         N =[n1, n0]
    %	s         standard deviation of the gaussian noise. Default is 0.5.
    %	wrappings number of wrappings of each spiral. Default is random.
    %	m 	  multiplier m of x * sin(m * x) for the second spiral. Default is random.
    %    OUTPUT
    %	X data matrix with a sample for each row 
    %   	Y vector with the labels
    %
    %   EXAMPLE:
    %       X, Y,_ ,_ ,_ = spiral([10, 10])
            X, Y, s, w, m = spiral([10, 10])
            X, Y, s, wrappings, m = spiral(N=[100,100], s = 0.5)
            X, y, s, wrappings, m = spiral(N=[100,100], s = 0.5,  wrappings =4.5, m = 3.2)
    '''
    if type(m)==str and m == 'random':
        m = 1 + np.random.rand()
    
    if type(wrappings)==str and wrappings =='random':
        wrappings = 1 + np.random.rand() * 8
        
    
    oneDSampling = np.random.rand(N[0], 1)*wrappings*np.pi
    
    x1 = np.hstack([np.multiply(oneDSampling,np.cos(oneDSampling)), np.multiply(oneDSampling,np.sin(oneDSampling))])
    
    x1 = x1 + np.random.randn(N[0], 2)*s
    
    
    oneDSampling = np.random.rand(N[1], 1)*wrappings*np.pi
    
    x2 = np.hstack([np.multiply(oneDSampling,np.cos(m*oneDSampling)), np.multiply(oneDSampling,np.sin(m*oneDSampling))])
    
    x2 = x2 + np.random.randn(N[1], 2)*s
    
    X = np.vstack([x1,x2])
    
    
    Y = np.ones([sum(N),1])
    Y[:N[0],0] = -1
        
    return X, Y, s, wrappings, m

def sinusoidal(N, s = 0.1):
    '''
    %Sample a dataset from a dataset separated by a sinusoidal line
    %   X, Y, s = sinusoidal(N, s)
    %    INPUT 
    %	N      1x2 vector that fix the numberof samples from each class
    % 	s      standard deviation of the gaussian noise. Default is 0.1
    %    OUTPUT
    %	X data matrix with a sample for each row 
    %   	Y vector with the labels
    %
    %   EXAMPLE:
    %       X, Y,_ = sinusoidal([10, 10])
    %       X, Y,_ = sinusoidal(N = [10, 10],s=0.5)
    '''
    X = np.array([0,0])
    while(X.shape[0]<=N[0]):
        xx = np.random.rand();
        yy = np.random.rand();
        fy = 0.7 * 0.5 * np.sin(2 * np.pi * xx) + 0.5;
        if(yy <= fy):
            xi = np.array([xx + s*np.random.rand(), yy + s*np.random.rand()])
            X = np.vstack([X, xi])
     
    X = np.delete(X,0,0)
    
    while(X.shape[0] < sum(N)):
        xx = np.random.rand();
        yy = np.random.rand();
        fy = 0.7 * 0.5 * np.sin(2 * np.pi * xx) + 0.5;
        if(yy > fy):
            xi = np.array([xx + s*np.random.rand(), yy + s*np.random.rand()])
            X = np.vstack([X, xi])
    
    Y = np.ones([sum(N),1])
    Y[:N[0],0] = -1
    
    return X, Y, s

def moons(N, s =0.1, d='random', angle = 'random'):
    '''
    %Sample a dataset from two "moon" distributions 
    %   X, Y, s, d, angle = moons(N, s, d, angle)
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
    %       X, Y,s,d,a = moons([10, 10])
            X, Y,_,_,_ = moons(N =[10, 10],s=0.5)
    '''
    if type(angle)==str and angle =='random':
        angle = np.random.rand() * np.pi
    
    if type(d)==str and d == 'random':
        d = (-0.6 * np.random.rand(1, 2) + np.array([-0.2, -0.2]))
    
    d1 = np.array([[np.cos(-angle),  -np.sin(-angle)],[np.sin(-angle),   np.cos(-angle)]])
    
    d = np.dot(d1,d.T).T[0]
    
    oneDSampling =  (np.pi + np.random.rand(1, N[0]) * 1.3 * np.pi + angle)[0]
    
    X = np.hstack([ np.array([np.sin(oneDSampling)]).T,   np.array([np.cos(oneDSampling)]).T])
    
    X = X + np.random.randn(N[0],2)*s
    
    
    oneDSampling =  (np.random.rand(1, N[1]) * 1.3 * np.pi + angle)[0]
    
    X1 = np.hstack([ np.array([np.sin(oneDSampling)]).T,   np.array([np.cos(oneDSampling)]).T])
    
    X1 = X1 + np.random.randn(N[1],2)*s + np.tile(d,(N[1],1))
    
    #[sin(oneDSampling.T) cos(oneDSampling')] + randn(N(2),2)*s + repmat(d, N(2), 1)
    #np.tile(d,(10,1))
    
    X = np.vstack([X,X1])
    
    Y = np.ones([sum(N),1])
    Y[:N[0],0] = -1
    
    return X, Y, s, d, angle

def gaussian(N, ndist = 3, means ='random', sigmas='random'):
    '''
    %Sample a dataset from a mixture of gaussians
    %   X, Y, ndist, means, sigmas = gaussian(N, ndist, means, sigmas)
    %    INPUT 
    %	N      1x2 vector that fix the numberof samples from each class
    %	ndist  number of gaussian for each class. Default is 3.    
    %	means  vector of size(2*ndist X 2) with the means of each gaussian. 
    %	       Default is random.
    %	sigmas A sequence of covariance matrices of size (2*ndist, 2). 
    %	       Default is random.
    %    OUTPUT
    %	X data matrix with a sample for each row 
    %   	Y vector with the labels
    %
    %   EXAMPLE:
    %       X, Y, ndist, means, sigmas = gaussian([10, 10])
            X, Y,_,_,_ = gaussian(N =[10, 10], ndist = 2)
    '''
    
    if type(sigmas)==str and sigmas == 'random':
        sigmas =[0,0]
        for i in range(ndist*2):
            sigma = np.random.rand(2, 2) + np.eye(2) * 2
            sigma[0,1] =sigma[1,0]
            sigmas = np.vstack([sigmas, sigma])
    
        sigmas = np.delete(sigmas,0,0)
    
    if type(means)==str and means == 'random':
        means = np.random.rand(ndist * 2, 2) * 20 - 10
            
    X = [0 , 0]
    
    for i in range(N[0]):
        dd = np.floor(np.random.rand() * ndist)
        dd = int(dd)
        xi = np.dot(np.random.randn(1,2),sigmas[dd*2:dd*2+2,:]) + means[dd, :]
        X = np.vstack([X,xi])
    
    X = np.delete(X,0,0)
    
    for i in range(N[1]):
        dd = np.floor(np.random.rand() * ndist + ndist)
        dd = int(dd)
        xi = np.dot(np.random.randn(1,2),sigmas[dd*2:dd*2+2,:]) + means[dd, :]
        X = np.vstack([X,xi])
    
    Y = np.ones([sum(N),1])
    Y[:N[0],0] = -1
    
    return X, Y, ndist, means, sigmas

def linear_data(N, m ='random', b ='random', s =0.1):
    '''
    %Sample a dataset from a linear separable dataset
    %   X, Y, m, b, s = linear(N, m, b)
    %    INPUT 
    %	N      1x2 vector that fix the numberof samples from each class
    %	m      slope of the separating line. Default is random.    
    %	b      bias of the line. Default is random.
    % 	s      standard deviation of the gaussian noise. Default is 0.1
    %    OUTPUT
    %	X data matrix with a sample for each row 
    %   	Y vector with the labels
    %
    %   EXAMPLE:
    %       X, Y, m, b, s = linearData([10, 10])
    %       X, Y, _, _,_ = linearData(N =[10, 10],s=0.5)
    '''
    
    if type(b) ==str and b == 'random':
        b = np.random.rand()*0.5;

    if type(m) ==str and m == 'random':
        m = np.random.rand() * 2 +0.01;
    
    
    X =np.array([0,0])
    
    while(X.shape[0]<=N[0]):
        xx = np.random.rand()
        yy = np.random.rand()
        fy = xx * m + b;
        if (yy<= fy):
            xi = [xx + np.random.randn()*s, yy + np.random.randn()*s]
            X = np.vstack([X,xi])
            
    X = np.delete(X,0,0)
    
    while(X.shape[0]<sum(N)):
        xx = np.random.rand()
        yy = np.random.rand()
        fy = xx * m + b;
        if (yy > fy):
            xi = [xx + np.random.randn()*s, yy + np.random.randn()*s]
            X = np.vstack([X,xi])
    
    Y = np.ones([sum(N),1])
    Y[:N[0],0] = -1
    
    return X, Y, m, b, s

def create_dataset(N, Dtype, noise, varargin = 'PRESET',**Options):
    '''    
    %Sample a dataset from different distributions
    %   [X, Y, varargout] = create_dataset(N, type, noise, varargin)
    %
    %   INPUT 
    %       N     Number of samples
    %       type  Type of distribution used. It must be one from 
    %            'MOONS' 'GAUSSIANS' 'LINEAR' 'SINUSOIDAL' 'SPIRAL'
    %       noise probability to have a wrong label in the dataset
    %	
    %       The meaning of the optional parameters depend on the type of the
    %       dataset, if is set to 'PRESET'a fixed set of parameters is used:
    %       'MOONS' parameters:
    %           1- s: standard deviation of the gaussian noise. Default is 0.1
    %           2- d: 1X2 translation vector between the two classes. With d = 0
    %                 the classes are placed on a circle. Default is random.
    %           3- angle: rotation angle of the moons in (radians). Default is random.
    %
    %       'GAUSSIANS' parameters:
    %           1- ndist: number of gaussians for each class. Default is 3.    
    %           2- means: vector of size(2*ndist X 2) with the means of each gaussian. 
    %              Default is random.
    %           3- sigmas: A sequence of covariance matrices of size (2*ndist, 2). 
    %              Default is random.
    %
    %       'LINEAR' parameters:
    %           1- m: slope of the separating line. Default is random.    
    %           2- b: bias of the line. Default is random.
    %           3- s: standard deviation of the gaussian noise. Default is 0.1
    %
    %       'SINUSOIDAL' parameters:
    %           1- s: standard deviation of the gaussian noise. Default is 0.1
    %
    %       'SPIRAL' parameters:
    %           1- s: standard deviation of the gaussian noise. Default is 0.5.
    %           2- wrappings: wrappings number of wrappings of each spiral. Default is random.
    %           3- m: multiplier m of x * sin(m * x) for the second spiral. Default is
    %                 random.
    %
    %  OUTPUT
    %   X data matrix with a sample for each row 
    %   Y vector with the labels
    %   varargout parameters used to sample data
    %   EXAMPLE:
    %       [X, Y] = create_dataset(100, 'SPIRAL', 0.01);
    %       [X, Y] = create_dataset(100, 'SPIRAL', 0.01, 'PRESET');
    %       [X, Y] = create_dataset(100, 'SPIRAL', 0, 0.1, 2, 2);
    %	[X, Y] = gaussian(NN, 2, [-5, -7; 2, -9; 10, 5; 12,-6], repmat(eye(2)* 3, 4, 1));
    '''
    Dtype = Dtype.upper()
    
    NN = [int(np.floor(N / 2.0)), int(np.ceil(N / 2.0))];
    
    usepreset = 0
    if varargin =='PRESET':
        usepreset = 1
        
    if Dtype =='MOONS':
        
        if usepreset == 1:
            X, Y, s, d, angle = moons(NN, s = 0.1, d = np.array([-0.5, -0.5]), angle = 0)  #s =0.1, d='random', angle = 'random'
        else:
            #Default Setting : moons(NN,s =0.1, d='random', angle = 'random')
            s = 0.1
            d = angle ='random'
            
            if Options.has_key('s'):
                s = Options['s']
            if Options.has_key('d'):
                d = Options['d']
            if Options.has_key('angle'):
                angle = Options['angle']
            
            X, Y, s, d, angle = moons(NN, s=s, d=d, angle=angle)
        
        varargout= [s, d, angle]
    
    elif Dtype=='GAUSSIANS':
        
        if usepreset == 1:
            # gaussian(N, ndist = 3, means ='random', sigmas='random')
            
            means1 = np.array([[-5, -7],[2, -9],[10, 5],[12,-6]])
            sigma1 = np.tile(np.eye(2)* 3, (4, 1))
            
            X, Y, ndist, means, sigmas = gaussian(NN, ndist =2, means = means1, sigmas = sigma1)
        else:
            #Default Setting : gaussian(N, ndist = 3, means ='random', sigmas='random')
            
            ndist = 3
            means = sigmas = 'random'
            
            if Options.has_key('ndist'):
                ndist = Options['ndist']
            
            if Options.has_key('means'):
                means = Options['means']
            
            if Options.has_key('sigmas'):
                sigmas = Options['sigmas']
            
            X, Y, ndist, means, sigmas = gaussian(NN, ndist = ndist, means = means, sigmas = sigma)
            
        varargout = [ndist, means, sigmas ]
    
    elif Dtype =='LINEAR':
        
        if usepreset == 1:
            # linear_data(N, m ='random', b ='random', s =0.1)
            X, Y, m, b, s = linear_data(NN, m = 1, b =0, s =0.1)
        else:
            #Default Setting : linear_data(N, m ='random', b ='random', s =0.1)
            
            s, m, b = 0.1, 'random','random'
            
            if Options.has_key('m'):
                m = Options['m']
            
            if Options.has_key('b'):
                b = Options['b']
            
            if Options.has_key('s'):
                s = Options['s']
            
            X, Y, m, b, s = linear_data(NN, m = m, b =b, s =s)
            
        varargout = [m, b, s]
    
    elif Dtype=='SINUSOIDAL':
        
        if usepreset == 1:
            # sinusoidal(N, s = 0.1)
            X, Y, s = sinusoidal(NN, s = 0.01)
        else:
            #Default Setting : sinusoidal(N, s = 0.1)
            s = 0.1
            if Options.has_key('s'):
                s = Options['s']
            
            X, Y, s = sinusoidal(NN, s = s)
            
        varargout = [s]
    
    elif Dtype =='SPIRAL':
        
        if usepreset == 1:
            # spiral(N, s = 0.5, wrappings = 'random', m = 'random')
            
            X, Y, s, wrappings, m = spiral(NN, s = 0.5, wrappings = 2, m = 2)
        else:
            # Default Setting :  spiral(N, s = 0.5, wrappings = 'random', m = 'random')
            
            s, wrappings, m = 0.5, 'random', 'random'
            
            if Options.has_key('s'):
                s = Options['s']
            
            if Options.has_key('wrappings'):
                wrappings = Options['wrappings']
            
            if Options.has_key('m'):
                m = Options['m']
            
            X, Y, s, wrappings, m = spiral(NN, s = s, wrappings = wrappings, m = m)
            
        varargout = [s, wrappings, m]
    
    else:
        
        tkMessageBox.showerror("Tips and tricks",'Specified dataset type is not correct. It must be one of MOONS, GAUSSIANS, LINEAR, SINUSOIDAL, SPIRAL')
        raise ValueError('Specified dataset type is not correct. It must be one of MOONS, GAUSSIANS, LINEAR, SINUSOIDAL, SPIRAL')
    
    
    swap = np.random.rand(Y.shape[0],1)<=noise
    Y[swap] = Y[swap]*-1

    return X, Y, varargout

def loadMatFile(filePath):
    import scipy.io
    mat = scipy.io.loadmat(filePath)
    if mat.has_key('x'):
        X = mat['x']
    elif mat.has_key('X'):
        X = mat['X']
    else:
        X = None
        print 'Input data x or X not found'
        
    if mat.has_key('y'):
        Y = mat['y']
    elif mat.has_key('Y'):
        y = mat['Y']
    else:
        Y = None
        print 'Target data y or Y not found'
     
    return X,Y,mat
    
def load_Dataset(fpath):
    import scipy.io
    mat = scipy.io.loadmat(fpath)
    x = mat['x']
    y = mat['y']
    xt = mat['xt']
    yt = mat['yt']
    return x,y,xt,yt

def plot_dataset(X, Y):
    '''
    %Plot a classifier and its train and test samples
    %   plot_dataset(X, Y)
    %   INPUT 
    %       X   data of the set (a sample for each row)
    %       Y   labels of the set
    '''
    plt.close()
    plt.plot(X[np.where(Y>=0)[0],0],X[np.where(Y>=0)[0],1],'.r')
    plt.plot(X[np.where(Y<0)[0],0],X[np.where(Y<0)[0],1],'.b')
    
    plt.xlim([np.min(X[:,0]),np.max(X[:,0])])
    plt.ylim([np.min(X[:,1]),np.max(X[:,1])])
    plt.show()
    
def plot_estimator(alpha, xTrain, yTrain, xTest, yTest, ker, par):
    '''
    %Plot a classifier and its train and test samples
    %   [] = plot_estimator(alpha, xTrain, yTrain, xTest, yTest, ker, par)
    %   INPUT 
    %       alpha        classifier solution
    %       xTrain       train samples
    %       yTrain       labels of the train samples
    %       xTest        test samples
    %       yTest        labels of the test samples
    %       ker          kernel of the classifier
    %       par          parameters of the kernel
    '''
    d =100
    
    min1 = np.min([xTrain[:,0],xTest[:,0]])
    max1 = np.max([xTrain[:,0],xTest[:,0]])
    
    min2 = np.min([xTrain[:,1],xTest[:,1]])
    max2 = np.max([xTrain[:,1],xTest[:,1]])
    
    X1  = np.linspace(min1,max1,d)
    X2  = np.linspace(min2,max2,d)
    
    Z = np.zeros(X1.shape[0],X2.shape[0])
    
    for i in X1:
        for j in X2:
            x_test = [X1[i], X2[j]]
            alphac = [[]]*1
            alphac = alpha
            pred   = classify(alphac, ker, par, xTrain, x_test)
            Z[j,i] = pred[0]
     
    
    plt.plot(xTest[np.where(yTest>=0)[0],0],xTest[np.where(yTest>=0)[0],1],'.b')
    plt.plot(xTest[np.where(yTest< 0)[0],0],xTest[np.where(yTest< 0)[0],1],'.r')
    

    plt.plot(xTrain[np.where(yTrain>=0)[0],0],xTrain[np.where(yTrain>=0)[0],1],'ob')
    plt.plot(xTrain[np.where(yTrain< 0)[0],0],xTrain[np.where(yTrain< 0)[0],1],'or')
    
    plt.contour(X1,X2,Z)
    
    plt.show()
