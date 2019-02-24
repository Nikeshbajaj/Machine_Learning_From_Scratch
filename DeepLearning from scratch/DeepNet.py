'''
Deep Neural Network  from scrach
V3.0

@Author _ Nikesh Bajaj
PhD Student at Queen Mary University of London &
University of Genova
Conact _ http://nikeshbajaj.in 
n.bajaj@qmul.ac.uk
bajaj.nikkey@gmail.com
'''


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class deepNet():
	def __init__(self,X,y,Xts =None, yts =None, Net = [2],NetAf =['tanh'], alpha=0.01,miniBatchSize = 0.1, printCostAt =100,
		     AdamOpt=False,B1=0.9,B2=0.99,lambd=0,keepProb =[1.0]):
		

		nx, m1 = X.shape
		ny, m2 = y.shape
		
		assert(m1 == m2)#,"X and y dimension mismatched")
		
		m = m1

		if len(Net)==0:
			NetAf=[]

		Class = np.unique(y)
		
		nC = len(Class)
		self.nC 	= nC

		if len(Net)>len(NetAf):
			NetAf = NetAf*len(Net)
		
		if nC>2:
			Af =  NetAf + ['softmax']
			yb = self.OneHotEncoding(y)
			ytsb = self.OneHotEncoding(yts)
			Network = [nx] + Net + [nC]
		else:
			Af =  NetAf + ['sig']
			yb = np.copy(y)
			ytsb = np.copy(yts)
			Network = [nx] + Net + [ny]

		if len(keepProb)==1:
			keepProb = [keepProb[0]]*(len(Network)-1) + [1.0]

		print("#Classes   : ", nC)
		print("#Features  : ", nx)
		print("#Examples  : ", m)
		print("Network    : ", Network)
		print("ActiFun    : ", Af)
		print("keepProb   : ", keepProb)

		self.X 		= X
		self.y 		= yb
		self.yo 	= y

		self.Xts 	= Xts
		self.yts 	= ytsb
		self.yto 	= yts



		self.Class 	=Class
		self.nC 	= nC
		self.nx 	= nx
		self.m  	= m
		self.alpha 	= alpha
		self.Network= Network
		self.Afun   = Af
		self.Para 	= self.InitNNx(self.Network,aFun=self.Afun,method ='xavier')
		self.cost  	= []
		self.epoCost= []
		self.tsCost= []
		self.Itr 	= 0
		self.AdamOpt= AdamOpt
		self.B1 	= B1
		self.B2 	= B2
		self.lambd  = lambd
		self.miniBatchSize=miniBatchSize
		self.printCostAt = printCostAt
		self.keepProb = keepProb

		#print(self.Para['W'])
		#print(len(self.Para['W']))

	def __str__(self):
		print('-------------Info---------------')
		print("#Classes   : ", self.nC)
		print("#Features  : ", self.nx)
		print("#Examples  : ", self.m)
		print("Network    : ", self.Network)
		print("ActiFun    : ", self.Afun)
		print("keepProb   : ", self.keepProb)
		print("Alpha      : ", self.alpha)
		print("B1, B2     : ", self.B1,self.B2)
		print("lambd      : ", self.lambd)
		print("AdamOpt    : ", self.AdamOpt)
		return '---------------------------'

	def fit(self,itr = 1, miniBatchSize = 'default'):
		if isinstance(miniBatchSize, str) and  miniBatchSize == 'default':
			miniBatchSize = self.miniBatchSize

		Xt, yt = self.MiniBatchCreation(self.X,self.y,miniBatchSize=miniBatchSize)

		for i in range(itr):
			self.Itr +=1

			iPara = self.InitNNx(self.Network,aFun=self.Afun,method ='zero')
			Vgrad = {"dW":iPara["W"], "db":iPara["b"]}
			iPara = self.InitNNx(self.Network,aFun=self.Afun,method ='zero')
			Sgrad = {"dW":iPara["W"], "db":iPara["b"]}

			for batch in range(len(Xt)):

				yp, ZADtemp  = self.FarwProp_drop(Xt[batch],self.Para,keepProb = self.keepProb)

				if self.nC>2:
					J = self.LogCost_mClass(yt[batch],yp)
				else:
					J = self.LogCost(yt[batch],yp)

				self.cost.append(J)
				#--Backward Propogation

				grads = self.BackProp_drop(Xt[batch],yt[batch],self.Para,ZADtemp,lambd=self.lambd,keepProb =self.keepProb)
				
				#--Update Parameters
				if self.AdamOpt:
					Vgrad,Sgrad = self.AdamGrad(Vgrad,Sgrad, grads, B1=self.B1, B2=self.B2)
					self.Para 	= self.AdamUpdate(self.Para, Vgrad,Sgrad,self.alpha, B1=self.B1, B2=self.B2, ith =batch+1)	
				else:
					self.Para   = self.UpdatePara(self.Para,grads,self.alpha)
			
			yp, ZADtemp = self.FarwProp(self.X,self.Para)#,keepProb = [1.0])		

			if self.nC>2:
				J = self.LogCost_mClass(self.y,yp)
			else:
				J = self.LogCost(self.y,yp)
			for jj in range(len(Xt)):
				self.epoCost.append(J)

			if self.Xts is not None:
				ytsp, tZADtemp = self.FarwProp(self.Xts,self.Para)

				if self.nC>2:
					Jt = self.LogCost_mClass(self.yts,ytsp)
				else:
					Jt = self.LogCost(self.yts,ytsp)
				for jj in range(len(Xt)):
					self.tsCost.append(Jt)


			if self.printCostAt>0 and self.Itr%self.printCostAt ==0:
				if self.Xts is None:
					print("Epoc @ %d : Cost %0.6e " %(self.Itr,J))
				else:
					print("Epoc @ %d : Training Cost %0.6e  Testing Cost %0.6e" %(self.Itr,J,Jt))

		return self.Para

	def predict(self,Xs):
		yp, ZADtemp = self.FarwProp(Xs,self.Para)
		if yp.shape[0]>1:
			yi = np.argmax(yp,axis=0).reshape([1,-1])
		else:
			yi = (yp>0.5)*1
		return yi,yp

	def PlotLCurve(self):
		plt.figure(1)
		plt.clf()
		itrr = np.linspace(1,self.Itr,len(self.cost))
		#if self.Xts is None:
		#	print('Epoc @ ',self.Itr, 'Cost : ',self.epoCost[-1])
		#else:
		#	print('Epoc @ ',self.Itr, 'Training Cost : ',self.epoCost[-1],' Testing Cost : ',self.tsCost[-1])
		plt.plot(itrr,self.cost,'--b',label ='Training(miniB)',alpha=0.2)
		plt.plot(itrr,self.epoCost,'r',label ='Training(fullB)')
		plt.plot(itrr,self.tsCost,'g',label ='Testing')
		plt.title("Learning Curve")
		plt.xlabel("Epoc")
		plt.ylabel("Cost")
		plt.legend()
		#plt.show()
		plt.pause(0.001)

	def PlotBoundries(self, X='default',y='default',Layers=True,density =300):
		if isinstance(X, str) and X == 'default':
			X = self.X
		if isinstance(y, str) and y == 'default':
			yi = self.y
			if self.y.shape[0]>1:
				yi = np.argmax(self.y,axis=0).reshape([1,-1])
		else:
			yi =y
		if X.shape[0]>2:
			print('Dimention of data is more than 2, boundries and layers could not be plotted')
		F = plt.figure(2,figsize=(8,4))
		plt.clf()
		#print(self.y.shape)
		
		if Layers:
			self.PlotDeepLayers2(X,yi,self.Para,Xts=self.Xts, yts=self.yts,prob=True, density =density)
		else:
			self.PlotDBoundries(X,yi,self.Para,Xts=self.Xts, yts=self.yts,prob=True, density =density)
		F.suptitle("Cost :" + str(np.around(self.cost[-1],2))+ " Epoc : "+str(self.Itr),y=1.0)
		#plt.show()
		plt.pause(0.001)
		
	def ActFun(self,z,fun='sig'):
		if fun=='sig':
			a = 1.0/(1+np.exp(-z))
		elif fun=='tanh':
	  		a = np.tanh(z)
		elif fun=='relu':
			a = z
			a[np.where(z<0)]=0
		elif fun=='lrelu':
			a = z
			a[z<0.01]=0.01
		
		elif fun=='softmax':
			a = self.softmax(z)
		else:
			a = None
			print("Activation Function does not match")
		return a

	def dActFun(self,z,fun='sig'):
		if fun=='sig':
			a = 1.0/(1+np.exp(-z))
			d = a*(1-a)
		elif fun=='tanh':
	  		a = np.tanh(z)
	  		d = 1-np.power(a,2)
		elif fun=='relu':
			#d = (z>0)*1.0
			d = np.zeros([z.shape[0],z.shape[1]])
			d[np.where(z>0)] = 1.0
		elif fun=='lrelu':
			d = np.ones([z.shape[0],z.shape[1]])*0.01
			d[np.where(z>0)] = 1.0
			#d[d==0]=0.01
		else:
			d = None
			print("Activation Function does not match")
		return d

	def softmax(self,z):
	    t = np.exp(z)
	    tsum = np.sum(t,axis = 0, keepdims = True)
	    a = t/tsum
	    return a

	def LogCost(self,y,yp):
		assert(y.shape == yp.shape)
		m = y.shape[1]
		if m==0:
			print("Cost dim : m :",m)
		logprobs = np.multiply(np.log(yp+1e-20),y) + np.multiply(np.log(1-yp+1e-20),1-y)
		J = - (1.0/m)*np.sum(logprobs)
		if np.isnan(J):
			logprobs = np.multiply(np.log(yp+1e-10),y) + np.multiply(np.log(1-yp+1e-10),1-y)
			J = - (1.0/m)*np.sum(logprobs)
		return J

	def LogCost_mClass(self,y,yp):
		#print("Shape",y.shape,yp.shape)
		assert(y.shape == yp.shape)
		m = y.shape[1]
		J = - (1.0/m)*np.sum(y*np.log(yp + 1e-10))
		return J

	def FarwProp(self,X,Para):
		n,m = X.shape
		W  = Para["W"]
		b  = Para["b"]
		Af = Para["aFun"]
		assert(len(W) == len(b))
		#assert(W[0].shape[1] ==n)
		if not(W[0].shape[1] ==n):
			print(W[0].shape, X.shape,n)
		A =[]
		Z =[]
		A.append(X)
		for l in range(len(W)):
			Zl = np.dot(W[l],A[l]) + b[l]
			Al = self.ActFun(Zl,fun=Af[l])
			Z.append(Zl)
			A.append(Al)

		ZAtemp ={"Z":Z,"A":A}

		return A[-1],ZAtemp

	def BackProp(self,X,y,Para,ZAtemp):
		#print("BackProp")
		n,m = X.shape
		
		W  = Para["W"]
		b  = Para["b"]
		Af = Para["aFun"]
		Z = ZAtemp["Z"]
		A = ZAtemp["A"]
		
		#print(W,b)
		dW = [""]*len(W)
		db = [""]*len(b)
		for l in range(len(W)-1,-1,-1):

		    if l<len(W)-1:
		    	#dZl = np.dot(W[l+1].T,dZl)*dActFun(A[l+1],Af[l])
		    	dZl = np.dot(W[l+1].T,dZl)*dActFun(Z[l],Af[l])
		    else:
		    	dZl = A[-1]-y

		    dWl = (1.0/m)*np.dot(dZl,A[l-0].T)
		    dbl = (1.0/m)*np.sum(dZl,axis=1,keepdims=True)
		    #print(dWl,dbl)
		    dW[l] = dWl
		    db[l] = dbl

		grads = {"dW":dW,"db":db}

		return grads

	def FarwProp_drop(self,X,Para,keepProb = 'default'):
		if isinstance(keepProb, str) and  keepProb == 'default':
			keepProb = self.keepProb

		n,m = X.shape
		W  = Para["W"]
		b  = Para["b"]
		Af = Para["aFun"]
		assert(len(W) == len(b))
		assert(W[0].shape[1] ==n)

		A =[]
		Z =[]
		
		A.append(X)

		if len(keepProb)==1:
			keepProb = [keepProb[0]]*(len(W)+1)

		assert(len(keepProb)==len(W)+1)
		D =[]  #dropouts list of matrix

		for l in range(len(W)):
			Dl    = np.random.rand(A[l].shape[0],A[l].shape[1])<keepProb[l]
			A[l] *= Dl
			A[l] /= keepProb[l]
			#print(A[l].shape,Dl.shape,keepProb[l])
			Zl    = np.dot(W[l],A[l]) + b[l]
			Al    = self.ActFun(Zl,fun=Af[l])
			Z.append(Zl)
			A.append(Al)
			D.append(Dl)

		ZADtemp ={"Z":Z,"A":A,"Drop":D}

		return A[-1],ZADtemp
	
	def BackProp_drop(self,X,y,Para,ZADtemp,lambd = 'default',keepProb = 'default'):
		if isinstance(lambd, str) and  lambd == 'default':
			lambd = self.lambd

		if isinstance(keepProb, str) and  keepProb == 'default':
			keepProb = self.keepProb


		n,m = X.shape
		W  = Para["W"]
		b  = Para["b"]
		Af = Para["aFun"]
		Z  = ZADtemp["Z"]
		A  = ZADtemp["A"]
		D  = ZADtemp["Drop"]

		dW = [""]*len(W)
		db = [""]*len(b)

		if len(keepProb)==1:
			keepProb = [keepProb[0]]*(len(W)+1)

		for l in range(len(W)-1,-1,-1):
		    if l<len(W)-1:
		    	# dZl = np.dot(W[l+1].T,dZl)*dActFun(Z[l],Af[l])
		    	dAl = np.dot(W[l+1].T,dZl)
		    	#print(dAl.shape,D[l-1].shape,D[l].shape,D[l+1].shape)
		    	#print(Z[l].shape)
		    	#print(dAl.shape,D[l+1].shape,Z[l].shape,keepProb[l+1])
		    	dAl *=D[l+1]
		    	dAl /=keepProb[l+1] 
		    	dZl = dAl*self.dActFun(Z[l],Af[l])
		    else:
		    	dZl = A[-1]-y

		    dWl = (1.0/m)*np.dot(dZl,A[l-0].T)
		    dbl = (1.0/m)*np.sum(dZl,axis=1,keepdims=True)
		    dW[l] = dWl + (lambd/float(m))*W[l]
		    db[l] = dbl

		grads = {"dW":dW,"db":db}

		return grads

	def UpdatePara(self,Para,grads,alpha = 'default'):
		if isinstance(alpha, str) and  alpha == 'default':
			alpha = self.alpha

		W  = Para["W"]
		b  = Para["b"]
		Af = Para["aFun"]

		dW = grads["dW"]
		db = grads["db"]

		for l in range(len(W)):
			W[l] = W[l] - alpha*dW[l]
			b[l] = b[l] - alpha*db[l]

		Para = {"W":W,"b":b,"aFun":Af}
		return Para

	def InitNNx(self,Network,aFun,method ='xavier'):
		W = []
		b = []
		for l in range(len(Network)-1):
			
			if method =='he':
				Var = np.sqrt(2.0/Network[l])
			elif method =='xavier':
				Var = np.sqrt(1.0/Network[l])
			elif method =='rand':
				Var = 0.01
			elif method =='zero':
				Var =0.0
			else:
				Var = 1.0

			Wl = np.random.randn(Network[l+1],Network[l])*Var
			if method =='zero':
				Wl = np.zeros([Network[l+1],Network[l]])
			
			bl = np.zeros([Network[l+1],1])
			W.append(Wl)
			b.append(bl)
		Para = {"W": W,"b": b,"aFun":aFun}
		return Para

	def AdamGrad(self,Vgrad,Sgrad,grad, B1='default',B2='default'):
		if isinstance(B1, str) and  B1 == 'default':
			B1 = self.B1
		if isinstance(B2, str) and  B2 == 'default':
			B2 = self.B2



		VdW = Vgrad["dW"]
		Vdb = Vgrad["db"]
		SdW = Sgrad["dW"]
		Sdb = Sgrad["db"]
		dW  = grad["dW"]
		db  = grad["db"]

		#print("ADAM---_GRAD")
		#print(Vgrad,Sgrad,grad)

		for i in range(len(dW)):
			VdW[i] = B1*VdW[i] + (1.0-B1)*dW[i]
			Vdb[i] = B1*Vdb[i] + (1.0-B1)*db[i]

			SdW[i] = B2*SdW[i] + (1.0-B2)*np.power(dW[i],2)
			Sdb[i] = B2*Sdb[i] + (1.0-B2)*np.power(db[i],2)


		Vgrad ={"dW":VdW,"db":Vdb}
		Sgrad ={"dW":SdW,"db":Sdb}

		#print("ADAM---_GRAD")
		#print(Vgrad,Sgrad,grad)

		return Vgrad,Sgrad

	def AdamUpdate(self, Para,Vgrad,Sgrad,alpha = 'default',B1='default',B2='default', ith = 100000,esp=1e-7,m=1,lambd = 'default'):
		if isinstance(B1, str) and  B1 == 'default':
			B1 = self.B1
		if isinstance(B2, str) and  B2 == 'default':
			B2 = self.B2
		if isinstance(lambd, str) and  lambd == 'default':
			lambd = self.lambd
		if isinstance(alpha, str) and  alpha == 'default':
			alpha = self.alpha

		W  = Para["W"]
		b  = Para["b"]
		Af = Para["aFun"]

		VdW = Vgrad["dW"]
		Vdb = Vgrad["db"]

		SdW = Sgrad["dW"]
		Sdb = Sgrad["db"]

		#print("ADAM---_Update")
		#print(Vgrad,Sgrad,Para)

		for l in range(len(W)):
			VdWc = VdW[l]/(1.0-(B1**ith))
			Vdbc = Vdb[l]/(1.0-(B1**ith))
			SdWc = np.sqrt(SdW[l]/(1.0-(B2**ith)) +esp)
			Sdbc = np.sqrt(Sdb[l]/(1.0-(B2**ith)) +esp)

			#print(VdWc,Vdbc,SdWc,Sdbc)

			W[l] = (1.0-alpha*lambd/m)*W[l] - alpha*(VdWc/SdWc)
			b[l] = (1.0-alpha*lambd/m)*b[l] - alpha*(Vdbc/Sdbc)

		Para = {"W":W,"b":b,"aFun":Af}
		return Para

	def MiniBatchCreation(self,X,y,miniBatchSize = 'default'):
		if isinstance(miniBatchSize, str) and  miniBatchSize == 'default':
			miniBatchSize = self.miniBatchSize


		nx,m1 = X.shape
		ny,m2 = y.shape
		assert(m1 == m2)
		m = m1

		ind = list(range(m))
		np.random.shuffle(ind)
		mSize  = int(np.floor(miniBatchSize*m))
		nBatch = int(m/mSize)
		#print("miniBatchSize :",m,mSize,nBatch)
		Xt = []
		yt = []

		for i in range(nBatch):
			if i==nBatch-1:
				inx = ind[i*mSize:]
			else:
				inx = ind[i*mSize:(i+1)*mSize]

			Xt.append(X[:,inx])
			yt.append(y[:,inx])

		return Xt,yt

	def OneHotEncoding(self,y, nCls=-1):
		assert(y.shape[0]==1)
		n,m=y.shape
		if nCls ==-1:
			#nC = len(np.unique(y))
			nC = self.nC
		else:
			nC = nCls
		print(n,m,nC)
		yb = np.zeros([nC,m])
		for i in range(m):
			yb[y[0,i],i]=1
		return yb

	def PlotDBoundries(self,X,y,Para,Xts=None,yts =None,prob=True, density =300):

		if not(prob):
			density =1000
		x1mn = np.min(X[0,:])
		x2mn = np.min(X[1,:])
		x1mx = np.max(X[0,:])
		x2mx = np.max(X[1,:])

		x1 = np.linspace(x1mn,x1mx,density)
		x2 = np.linspace(x2mn,x2mx,density)
		xv, yv = np.meshgrid(x1, x2)
		Xall = np.vstack([xv.flatten(),yv.flatten()])
		
		yp,_ = self.FarwProp(Xall,Para)

		#print(yp.shape)

		if yp.shape[0]>1:
			
			ypi = np.argmax(yp,axis=0)

			#ypj = np.max(yp,axis=0)

			#ypi = ypi+ypj

			ypi = ypi.reshape(density,density)

			#ypi = ypj.reshape(density,density)

			plt.imshow(ypi, origin='lower', interpolation='bicubic', 
					extent=(xv.min(), xv.max(), yv.min(), yv.max()))
			

			for i in np.unique(y):
				in0 = np.where(y==i)[1]
				plt.plot(X[0,in0],X[1,in0],'*')
				plt.tight_layout()

		else:
			yp = yp.reshape(density,density)

			if prob:
				plt.imshow(yp, origin='lower', interpolation='bicubic', 
					extent=(xv.min(), xv.max(), yv.min(), yv.max()))
			else:
				plt.imshow(yp>=0.5, origin='lower', interpolation='bicubic', 
					extent=(xv.min(), xv.max(), yv.min(), yv.max()))
			
			in0 = np.where(y==0)[1]
			in1 = np.where(y==1)[1]
			plt.plot(X[0,in0],X[1,in0],'*b')
			plt.plot(X[0,in1],X[1,in1],'*y')
			plt.tight_layout()

		return None

	def PlotDeepLayers(self,X,y,Para,prob=True, density =300):

		if not(prob):
			density =1000
		x1mn = np.min(X[0,:])
		x2mn = np.min(X[1,:])
		x1mx = np.max(X[0,:])
		x2mx = np.max(X[1,:])

		x1 = np.linspace(x1mn,x1mx,density)
		x2 = np.linspace(x2mn,x2mx,density)
		xv, yv = np.meshgrid(x1, x2)
		Xall = np.vstack([xv.flatten(),yv.flatten()])

		yp,ZAtemp = self.FarwProp(Xall,Para)

		A = ZAtemp["A"]
		
		Net =[]
		for i in range(len(A)):
			Net.append(A[i].shape[0])


		#print("Net :",Net)
		ncol = len(Net)
		nrow = max(Net)

		l = np.array(list(range(1,ncol*nrow+1))).reshape(nrow,ncol)
		l = l.T
		l = np.reshape(l,-1)

		k=1
		for i in range(len(A)):
			#print(A[i].shape)
			Ai = A[i]
			k=nrow*i+1
			#print("-",k)
			for j in range(Ai.shape[0]):
				yi = Ai[j,:].reshape([density,density])
				#print(k)
				#print(l[k-1])
				plt.subplot(nrow,ncol,l[k-1])
				k = k+1
				plt.imshow(yi,origin='lower', interpolation='bicubic',extent=(xv.min(), xv.max(), yv.min(), yv.max()))
		#plt.show()

	def PlotDeepLayers2(self,X,y,Para,Xts=None,yts =None,prob=True,density =300):

		if not(prob):
			density =1000
		if Xts is None:
			x1mn = np.min(X[0,:])
			x2mn = np.min(X[1,:])
			x1mx = np.max(X[0,:])
			x2mx = np.max(X[1,:])
		else:
			x1mn = np.min(np.hstack([X[0,:],Xts[0,:]]))
			x2mn = np.min(np.hstack([X[1,:],Xts[1,:]]))
			x1mx = np.max(np.hstack([X[0,:],Xts[0,:]]))
			x2mx = np.max(np.hstack([X[1,:],Xts[1,:]]))


		x1 = np.linspace(x1mn,x1mx,density)
		x2 = np.linspace(x2mn,x2mx,density)
		xv, yv = np.meshgrid(x1, x2)
		Xall = np.vstack([xv.flatten(),yv.flatten()])

		
		#Xall = np.vstack([Xall,Xall**2])
		#Xall = np.vstack([Xall,Xall**2,np.sqrt(Xall+10),Xall**3])

		#print('NaN :',sum(sum(np.isnan(Xall))))
		yp,ZAtemp = self.FarwProp(Xall,Para)

		A = ZAtemp["A"]
		
		Net =[]
		for i in range(len(A)):
			Net.append(A[i].shape[0])

		ncol = len(Net) + 1
		nrow = max(Net)

		l = np.array(list(range(1,ncol*nrow+1))).reshape(nrow,ncol)
		l = l.T
		l = np.reshape(l,-1)


		mrow = nrow//2 -1

		yi = A[0][0,:].reshape([density,density])
		plt.subplot(1,ncol,1)
		plt.subplots_adjust(hspace = .1, wspace=None)
		#plt.subplots_adjust(vspace = .1)
		plt.imshow(yi*0+1,origin='lower', interpolation='bicubic',extent=(xv.min(), xv.max(), yv.min(), yv.max()), alpha=0.1)
		for i in np.unique(y):
			in0 = np.where(y==i)[1]
			plt.plot(X[0,in0],X[1,in0],'.')
			#plt.axis('off')
			#plt.axis('equal')
			#plt.tight_layout()
		plt.title('Input Data')
		#plt.xlabel('Input Data')

		k=1
		for i in range(1,len(A)):
			#print(A[i].shape)
			Ai = A[i]
			k=nrow*i+1
			#print("-",k)
			for j in range(Ai.shape[0]):
				yi = Ai[j,:].reshape([density,density])
				plt.subplot(nrow,ncol,l[k-1])
				plt.subplots_adjust(hspace = .1, wspace=0.0)
				k = k+1
				plt.imshow(yi,origin='lower', interpolation='bicubic',extent=(xv.min(), xv.max(), yv.min(), yv.max()))
				plt.axis('off')
				if j==0:
					plt.title('Layer ' +str(i+1))


		k=nrow*(i+1)+1
		#plt.subplot(nrow,ncol,l[k-1 +mrow])
		plt.subplot(1,ncol,ncol)
		plt.subplots_adjust(hspace = .1, wspace=0.01)
		if yp.shape[0]>1:
			ypi = np.argmax(yp,axis=0)
			ypi = ypi.reshape(density,density)
			plt.imshow(ypi, origin='lower', interpolation='bicubic', 
					extent=(xv.min(), xv.max(), yv.min(), yv.max()))
			plt.axis('off')
			for i in np.unique(y):
				in0 = np.where(y==i)[1]
				plt.plot(X[0,in0],X[1,in0],'.')
				#plt.tight_layout()
		else:
			yp = yp.reshape(density,density)
			if prob:
				plt.imshow(yp, origin='lower', interpolation='bicubic', 
					extent=(xv.min(), xv.max(), yv.min(), yv.max()))
			else:
				plt.imshow(yp>=0.5, origin='lower', interpolation='bicubic', 
					extent=(xv.min(), xv.max(), yv.min(), yv.max()))
			
			in0 = np.where(y==0)[1]
			in1 = np.where(y==1)[1]
			plt.plot(X[0,in0],X[1,in0],'.b',label='Training')
			plt.plot(X[0,in1],X[1,in1],'.y')

			
			if Xts is not None:
				in0 = np.where(yts==0)[1]
				in1 = np.where(yts==1)[1]
				plt.plot(Xts[0,in0],Xts[1,in0],'*b',label='Testing')
				plt.plot(Xts[0,in1],Xts[1,in1],'*y')
				plt.legend(bbox_to_anchor=(0, 1.1), loc=3,borderaxespad=0., fontsize=7)

			#plt.tight_layout()
		#plt.show()



