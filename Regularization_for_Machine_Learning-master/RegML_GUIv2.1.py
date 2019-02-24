from Tkinter import *
import tkFont
import ttk
from tkFileDialog import askopenfilename
import tkMessageBox

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import RegML as rg


class RegML:
    def __init__(self,root):
        root.title("Regularization for Machine Learning")
        root.resizable(width =False,height =False)
        Main = ttk.Frame(root, padding=(3,3,12,12))#,width=600, height=600)
        f1 = ttk.Labelframe(Main, text = 'Input',borderwidth=1,relief="sunken",width=300, height=300 ,padding=(10,10,12,12))#
        f2 = ttk.Frame(Main, borderwidth=1, relief="sunken", width=300, height=400,padding=(10,10,12,12))
        self.f3 = ttk.Frame(Main, borderwidth=5, relief="sunken", width=300, height=400,padding=(1,1,1,1))
        f4 = ttk.Frame(Main, borderwidth=1, relief="sunken", width=300, height=300,padding=(3,3,3,3))

        f11 = ttk.Labelframe(f1, text = 'Task', borderwidth=1,relief="sunken",width=300, height=100,padding=(3,3,12,12))#
        f12 = ttk.Labelframe(f1, text = 'Dataset', borderwidth=1,relief="sunken",width=300, height=200,padding=(3,3,20,20))#

        f21 = ttk.Labelframe(f2, text = 'Filter',borderwidth=1, relief="sunken", width=300, height=50,padding=(3,3,12,12))
        f22 = ttk.Labelframe(f2, text = 'Kernal',borderwidth=1, relief="sunken", width=300, height=100,padding=(3,3,12,12))
        f23 = ttk.Labelframe(f2, text = 'Learning',borderwidth=1, relief="sunken", width=300, height=200,padding=(3,3,12,12))
        
        f41 = ttk.Labelframe(f4, text = 'Result',borderwidth=1, relief="sunken", width=400, height=70,padding=(3,3,1,1))
        self.f42 = ttk.Labelframe(f4, borderwidth=5, relief="sunken", width=400, height=220,padding=(0,0,0,0))


        Main.grid(column=0, row=0, sticky=(N, S, E, W))

        f1.grid(column=0, row=0, columnspan=1, rowspan=1, sticky=(N, S, E, W))
        f2.grid(column=1, row=0, columnspan=3, rowspan=2, sticky=(N, S, E, W))
        self.f3.grid(column=0, row=1, columnspan=1, rowspan=2, sticky=(N, S, E, W))
        f4.grid(column=1, row=2, columnspan=3, rowspan=1, sticky=(N, S, E, W))

        f11.grid(column=0, row=1, columnspan=1, rowspan=1, sticky=(N, S, E, W),pady=5)
        f12.grid(column=0, row=2, columnspan=1, rowspan=1, sticky=(N, S, E, W),pady=5)

        f21.grid(column=0, row=0, columnspan=1, rowspan=1, sticky=(N, S, E, W),pady=5)
        f22.grid(column=0, row=1, columnspan=1, rowspan=1, sticky=(N, S, E, W),pady=5)
        f23.grid(column=0, row=2, columnspan=1, rowspan=1, sticky=(N, S, E, W),pady=5)
        
        f41.grid(row=0, sticky=(N, S, E, W),pady=5)
        self.f42.grid(row=1, sticky=(N, S))

        ###-------------------------------- F1------------------------------------
        label1f1 = ttk.Label(f1, text='Input ')
        label2f1 = ttk.Label(f11, text='Task')
        label3f1 = ttk.Label(f12, text='Dataset')


        tFont = tkFont.Font(f12,family='Times', weight='bold') #, size=12

        label4f1 = ttk.Label(f12, text='# Training Samples')#,style ="B.TLabel")#,font =tFont)
        label5f1 = ttk.Label(f12, text='# Test Samples')
        label6f1 = ttk.Label(f12, text='Wrong labels ratio[0-1]')

        labelfont = ('timesroman', 15, 'bold')
        
        self.task = IntVar()
        self.task.set(1)
        self.Clf = ttk.Radiobutton(f11, text='Claasification', variable=self.task, value = 1,command=self.TaskSelF)
        self.Reg = ttk.Radiobutton(f11, text='Regression   ', variable=self.task, value =0,command=self.TaskSelF, state='disabled')
        
        self.dt = StringVar()
        self.eData = ttk.Radiobutton(f12, text='Existing Dataset', variable=self.dt, value=1, command=self.seDataFun)
        self.sData = ttk.Radiobutton(f12, text='Simulation   ', variable=self.dt, value=0, command=self.seDataFun)#, state='disabled')

        self.Browse = ttk.Button(f12, text="Browse", command=self.BrowseDataFun)
        #self.eDataO = StringVar()
        self.eDataG = ttk.Entry(f12, width = 8) 

        self.sDataG = ttk.Combobox(f12)#, textvariable = self.sDataO)
        self.sDataG['values'] = ('Moons', 'Gaussians','Linear','Sinusoidal', 'Spiral')#,'Toy')
                                 #MOONS, GAUSSIANS, LINEAR, SINUSOIDAL, SPIRAL
        self.sDataG.current(0)
        self.nTr = StringVar()
        self.nTs = StringVar()
        self.nE = StringVar()
        


        self.nTrain = ttk.Entry(f12, textvariable=self.nTr, width = 7)
        self.nTest = ttk.Entry(f12, textvariable=self.nTs, width = 7)
        self.nWL = ttk.Entry(f12, textvariable=self.nE, width = 7)

        self.nTrain.insert(0, '100')
        self.nTest.insert(0, '100')
        self.nWL.insert(0, '0.0')
        

        self.LoadData = ttk.Button(f12, text="LoadData",command = self.LoadDataFun)
        
        ## --------------F1----------Grid--------------------------------
        #F11
        self.Clf.grid(column=0, row=2,columnspan=2, rowspan=1, sticky=(N, S, E, W))
        self.Reg.grid(column=3, row=2,columnspan=1, rowspan=1, sticky=(N, S, E, W))

        #F12
        self.eData.grid(column=0, row=1,columnspan=2, rowspan=1, sticky=(N, S, E, W))
        self.eDataG.grid(column=2, row=1,columnspan=1, rowspan=1, sticky=(N, S, E, W),pady=5)
        self.Browse.grid(column=3, row=1,columnspan=2, rowspan=1, sticky=(S, W), padx=10)
        self.sData.grid(column=0, row=2,columnspan=2, rowspan=1, sticky=(N, S, E, W))
        self.sDataG.grid(column=2, row=2,columnspan=1, rowspan=1, sticky=(N, S, E, W))

        label4f1.grid(column=2, row=3,columnspan=1, rowspan=1, sticky=(N, S, E, W))
        label5f1.grid(column=2, row=4,columnspan=1, rowspan=1, sticky=(N, S, E, W))
        label6f1.grid(column=2, row=5,columnspan=1, rowspan=1, sticky=(N, S, E, W))
        
        self.nTrain.grid(column=3, row=3,columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=10,pady=2)
        self.nTest.grid(column=3, row=4,columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=10,pady=2)
        self.nWL.grid(column=3, row=5,columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=10,pady=2)
        self.LoadData.grid(column=0, row=6,columnspan=1, rowspan=1, sticky=(N, S, E, W))


        ###-------------------------------- F2------------------------------------
        #Filter------
        label1f2 = ttk.Label(f21, text='Filter')
        # DropDown
        
        self.ftrO = ttk.Combobox(f21,width=30)
        self.ftrO['values'] = ('Regularized Least Squares', 'nu-method', 'Truncated SVD','Spectral cut-off','Iterative Landweber')
        self.ftrO.current(0)
        self.ftrO.bind("<<ComboboxSelected>>", self.filtrUpdate)
        #Kernal------
        label2f2 = ttk.Label(f22, text='Kernal')
        # DropDown
        self.krnlO = ttk.Combobox(f22)
        self.krnlO['values'] = ('Linear', 'Polynomial', 'Gaussian')
        self.krnlO.current(0)
        self.krnlO.bind("<<ComboboxSelected>>", self.kpRangeUpdate)

        
        #self.kparVar = StringVar()
        self.kParaO = ttk.Scale(f22, orient=HORIZONTAL, length=200)#, from_=0.0, to=100.0)
        self.kParaO['command'] = self.kParaSelFun
        self.kParaO['from_']= 1.0
        self.kParaO['to']= 30.0
        
        self.label3f2 = ttk.Label(f22, text='kpar : 0.0')
        self.aSigVar = IntVar()
        self.aSigmaO = ttk.Checkbutton(f22, text='AutoSigma', variable = self.aSigVar,command=self.aSigmaSelFun)
        
        ukpar = DoubleVar()
        self.ukparO = ttk.Entry(f22, width = 8) #textvariable=ukpar,
        self.ukparO.insert(0,0.0)

        label3f2 = ttk.Label(f23, text='Learning')
        
        self.tVar = IntVar()
        self.cKvcO = ttk.Checkbutton(f23, text='use KCV',variable = self.tVar,onvalue =1, offvalue =0,command=self.cKCVSelfun)
        self.fixedVO = ttk.Checkbutton(f23, text='Fixed Value',variable = self.tVar,onvalue =0, offvalue =1,command=self.cKCVSelfun)

        label31f2 = ttk.Label(f23, text='Split')
        label32f2 = ttk.Label(f23, text='# split')
        label33f2 = ttk.Label(f23, text='t min')
        label34f2 = ttk.Label(f23, text='t max')
        label35f2 = ttk.Label(f23, text='# of values')
        
        # DropDown1
        #splitype = StringVar()
        self.splitO = ttk.Combobox(f23)#, textvariable = splitype)
        self.splitO['values'] = ('Sequential', 'Random')
        self.splitO.current(0)

        # DropDown2
        
        self.vScaleO = ttk.Combobox(f23)
        self.vScaleO['values'] = ('Linear Space', 'Log Space')
        self.vScaleO.current(0)

        self.nSO   = ttk.Entry(f23, width = 4) 
        self.tMnO  = ttk.Entry(f23, width = 4) 
        self.tMxO  = ttk.Entry(f23, width = 4) 
        self.nSvO  = ttk.Entry(f23, width = 4) 
        self.fixVO = ttk.Entry(f23, width = 4)

        self.nSO.insert(0,5)
        self.tMnO.insert(0,0.0)
        self.tMxO.insert(0,1.0)
        self.nSvO.insert(0,10)
        self.fixVO.insert(0,1.3)


        ## --------------F2----------Grid--------------------------------
        ##F21
        self.ftrO.grid(column=0, row=0,columnspan=2, rowspan=1, sticky=(N, S, E, W),padx=10)

        ## F22
        #Kernal------
        # DropDown
        self.krnlO.grid(column=1, row=1,columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=10,pady=5)
        self.kParaO.grid(column=1, row=2,columnspan=2, rowspan=1, sticky=(N, S, E, W),padx=10,pady=5)
        self.label3f2.grid(column=0, row=2,columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=5,pady=5)
        self.aSigmaO.grid(column=0, row=3,columnspan=1, rowspan=1, sticky=(N, S, E, W))
        self.ukparO.grid(column=2, row=3,columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=10,pady=2)
        

        ## F23
        self.cKvcO.grid(column=0, row=1,columnspan=1, rowspan=1, sticky=(N, S, E, W))
        label31f2.grid(column=0, row=2,columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=15)
        label32f2.grid(column=0, row=3,columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=15)
        label33f2.grid(column=0, row=4,columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=15)
        label34f2.grid(column=0, row=5,columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=15)
        label35f2.grid(column=0, row=6,columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=15)
        # DropDown2
        self.vScaleO.grid(column=0, row=7,columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=20)
        self.fixedVO.grid(column=0, row=8,columnspan=1, rowspan=1, sticky=(N, S, E, W))
        # DropDown1
        self.splitO.grid(column=1, row=2,columnspan=2, rowspan=1, sticky=(N, S, E, W),pady=4)
        #Entry--
        self.nSO.grid(column=2, row=3,columnspan=1, rowspan=1, sticky=(N, S, E, W),pady=2)
        self.tMnO.grid(column=2, row=4,columnspan=1, rowspan=1, sticky=(N, S, E, W),pady=2)
        self.tMxO.grid(column=2, row=5,columnspan=1, rowspan=1, sticky=(N, S, E, W),pady=2)
        self.nSvO.grid(column=2, row=6,columnspan=1, rowspan=1, sticky=(N, S, E, W),pady=2)
        self.fixVO.grid(column=1, row=8,columnspan=1, rowspan=1, sticky=(N, S, E, W),pady=2)

        ###-------------------------------- F3------------------------------------
        ## Plot
        
        ###-------------------------------- F4------------------------------------
        #F4
        self.label1f41 = ttk.Label(f41, text='Training Error  - ')
        self.label2f41 = ttk.Label(f41, text='Testing Error   - ')
        self.label3f41 = ttk.Label(f41, text='Selected  t  - ')
        self.sep1f41   = ttk.Separator(f41, orient="vertical")

        ##--------------- F4---------Grid---------------------------
        self.label1f41.grid(column=0, row=0, columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=10,pady=5)
        self.label2f41.grid(column=0, row=1, columnspan=1, rowspan=1, sticky=(N, S, E, W),padx=10,pady=5)
        self.sep1f41.grid(column=1, row=0, columnspan=1, rowspan=3, sticky='NS',padx=10,pady=0)
        self.label3f41.grid(column=2, row=0, columnspan=1, rowspan=2, sticky=(N, S, E, W),padx=10,pady=5)
               
        
        #######----------MAIN Frame----------------------------------------------------------
        
        Plot = ttk.Button(Main, text="Plot Test/Train", command = self.PlotTrTsTT)
        RUN = ttk.Button(Main, text="RUN" ,command = self.runLEARN)
        CLOSE = ttk.Button(Main, text="Close",command = self.Qt)

        Reset = ttk.Button(Main, text="Reset",command = self.ResetSetting)

        
        Plot.grid(column=0, row=3)
        RUN.grid(column=1, row=3)
        Reset.grid(column=2, row=3)        
        CLOSE.grid(column=3, row=3)

        #---------Default--Initial Setting variable/Items-------------------
        self.plotTrain = True
        self.trained = False
        self.Clf.invoke()
        self.sData.invoke()
        self.aSigmaO.state(['selected'])
        self.kParaO.state(['disabled'])
        self.fixedVO.state(['selected'])
        self.fixedVO.invoke()
        self.ftrO.current(0)

    ##---------------------------------Function--------------------------------------
    def Qt(self):
        root.quit()
        root.destroy()
        
    def ResetSetting(self):
        self.nTrain.delete(0, 'end')
        self.nTest.delete(0, 'end')
        self.nWL.delete(0, 'end')
        self.nSO.delete(0, 'end')
        self.tMnO.delete(0, 'end')
        self.tMxO.delete(0, 'end')
        self.nSvO.delete(0, 'end')
        self.fixVO.delete(0, 'end')
        self.ukparO.delete(0, 'end')

        self.nTrain.insert(0, '100')
        self.nTest.insert(0, '100')
        self.nWL.insert(0, '0.01')
        self.nSO.insert(0,5)
        self.tMnO.insert(0,0.0)
        self.tMxO.insert(0,1.0)
        self.nSvO.insert(0,10)
        self.fixVO.insert(0,1.3)
        self.ukparO.insert(0,0.0)

        self.kParaO['from_']= 1.0
        self.kParaO['to']= 30.0
        
        self.task.set(1)
        
        self.sDataG.current(0)
        self.ftrO.current(0)
        self.krnlO.current(0)
        self.vScaleO.current(0)
        self.splitO.current(0)

        self.aSigmaO.state(['selected'])
        self.kParaO.state(['disabled'])
        self.fixedVO.state(['selected'])
        

        self.fixedVO.invoke()
        self.Clf.invoke()
        self.sData.invoke()
        self.plotTrain = True
        self.trained = False

    def TaskSelF(self):
        if self.Clf.instate(['active']):
            print 'CLF is instate active'
            
        if 'active' in self.Clf.state():
            #print 'CLF is active'
            self.task.set(1)
        elif 'selected' in self.Clf.state():
            #print 'CLF is selected'
            self.task.set(1)
        if 'active' in self.Reg.state():
            #print 'Reg is active'
            self.task.set(0)
        elif 'selected' in self.Reg.state():
            #print 'Reg is selected'
            self.task.set(0)
        
    def seDataFun(self):
        if self.eData.instate(['active']):
            print 'eData is instate active'
            
        if 'active' in self.eData.state():
            self.dt.set(1)
            self.sDataG.state(['disabled'])
            self.nTrain.state(['disabled'])
            self.nTest.state(['disabled'])
            self.nWL.state(['disabled'])
            self.LoadData.state(['disabled'])
            self.Browse.state(['!disabled'])
            
        elif 'selected' in self.eData.state():
            #print 'eData is active'
            self.dt.set(1)
            self.sDataG.state(['disabled'])
            self.nTrain.state(['disabled'])
            self.nTest.state(['disabled'])
            self.nWL.state(['disabled'])
            self.LoadData.state(['disabled'])
            self.Browse.state(['!disabled'])
            
        if 'active' in self.sData.state():
            #print 'sData is active'
            self.dt.set(0)
            #self.sDataG.state(['active'])
            self.sDataG.state(['!disabled'])
            self.nTrain.state(['!disabled'])
            self.nTest.state(['!disabled'])
            self.nWL.state(['!disabled'])
            self.LoadData.state(['!disabled'])
            self.Browse.state(['disabled'])
            
            
        elif 'selected' in self.sData.state():
            #print 'sData is active'
            self.dt.set(0)
            #self.sDataG.state(['active'])
            self.sDataG.state(['!disabled'])
            self.nTrain.state(['!disabled'])
            self.nTest.state(['!disabled'])
            self.nWL.state(['!disabled'])
            self.LoadData.state(['!disabled'])
            self.Browse.state(['disabled'])
        
    def BrowseDataFun(self):
        fpath = askopenfilename(filetypes = (("Matlab files", "*.mat") ,("CSV Files", "*.csv")))
        print fpath
        
        if fpath[-4:] =='.mat':
            print 'MATLAB File'       
            self.X, self.y, self.Xt, self.yt = rg.load_Dataset(fpath)
        elif fpath[-4:] =='.csv':
            print 'CSV File ........... Coming Soon'

        #self.eDataG.config['text'] = fpath
        self.eDataG.delete(0, 'end')
        self.eDataG.insert(0, fpath)
        self.Dtype = 'Loaded'
        self.plotTrain = True
        self.PlotTrTsTT()
    
    def LoadDataFun(self):
        self.N =[int(self.nTrain.get()), int(self.nTest.get())]
        self.noise = float(self.nWL.get())
        self.Dtype = self.sDataG.get()
        print self.N, self.noise
        print 'Dtype :' , self.sDataG.get()
        
        #Options = {'s':0.1,'m1':2,'d':np.array([0.1, 0.1]), 'ndist':3}
        #p = self.noise/float(self.nTrain.get())
        self.X, self.y ,Op = rg.create_dataset(N = self.N[0], Dtype=self.Dtype, noise = self.noise, varargin = 'PRESET')
        
        #p = self.noise/float(self.nTest.get())
        self.Xt, self.yt,Op = rg.create_dataset(N = self.N[1], Dtype=self.Dtype, noise = self.noise, varargin = 'PRESET')
        
        print 'Shape : ',self.X.shape, self.y.shape, self.Xt.shape, self.yt.shape
        self.plotTrain = True
        self.trained = False
        #self.PlotTrTs()
        self.PlotTrTsTT()
        #self.PlotTrTsTT_New()
        #self.PlotTrTsF3()
    
    def PlotTrTsTT(self):
        f3f = Figure(figsize=(4.5, 3.5), dpi=100)
        a = f3f.add_subplot(111)
        if self.trained:
            a.plot(self.c[np.where(self.cyt>=0)[0],0],self.c[np.where(self.cyt>=0)[0],1],'.r' ,alpha=0.1)
            a.plot(self.c[np.where(self.cyt< 0)[0],0],self.c[np.where(self.cyt< 0)[0],1],'.b' ,alpha=0.1)
            
        if (self.plotTrain):
            self.plotTrain = False
            X = self.X
            Y = self.y
            a.set_title('Training Data')
        else:
            self.plotTrain = True
            X = self.Xt
            Y = self.yt
            a.set_title('Test Data')
        
        a.plot(X[np.where(Y>=0)[0],0],X[np.where(Y>=0)[0],1],'*r')
        a.plot(X[np.where(Y<0)[0],0],X[np.where(Y<0)[0],1],'*b')
        x1mx0 = [np.min(X[:,0]),np.max(X[:,0])]
        y1mx0 = [np.min(X[:,1]),np.max(X[:,1])]

        x0mn = np.min([np.min(self.X[:,0]),np.min(self.Xt[:,0])])
        x0mx = np.max([np.max(self.X[:,0]),np.max(self.Xt[:,0])])
        
        x1mn = np.min([np.min(self.X[:,1]),np.min(self.Xt[:,1])])
        x1mx = np.max([np.max(self.X[:,1]),np.max(self.Xt[:,1])])

        a.set_xlim([x0mn,x0mx])
        a.set_ylim([x1mn,x1mx])
        f3f.tight_layout()
        canvas = FigureCanvasTkAgg(f3f, master=self.f3)
        canvas.show()
        canvas.get_tk_widget().grid(row=0)

        #toolbar = NavigationToolbar2TkAgg(canvas, self.f3)
        #toolbar.grid(row=1)
        #toolbar.update()
        canvas._tkcanvas.grid()

    def PlotKCVErro(self):
        #self.avg_err_kcv = avg_err_kcv
        #self.t_kcv_idx   = t_kcv_idx
        #self.vnmx = [vmin,vmax]
        if self.cKvcO.instate(['selected']):
            f4f = Figure(figsize=(4.5, 2.2), dpi=85)
            a = f4f.add_subplot(111)
            if(len(list(self.trange)) > 1):
                a.plot(self.trange, self.avg_err_kcv,'-.b')
                a.plot(self.trange[self.t_kcv_idx], self.avg_err_kcv[self.t_kcv_idx], '*r')
                a.set_xlabel('reg.par. range: ' + str(self.vnmx[0]) + ' - ' + str(self.vnmx[1]),fontsize = 10)
            else:
                a.plot(range(1, len(list(self.avg_err_kcv))+1),self.avg_err_kcv,'-.b')
                a.plot(self.t_kcv_idx +1, self.avg_err_kcv[self.t_kcv_idx], '*r');
                a.set_xlabel('t range: ' + str(self.vnmx[0]) + ' - ' + str(self.vnmx[1]) ,fontsize = 10)
                
            a.set_title('KCV - Error Plot',fontsize = 12)
            a.set_ylabel('Error')#,fontsize = 10)
            a.grid(True)    
            a.margins(0.1, 0.1)
            f4f.tight_layout()
            canvas = FigureCanvasTkAgg(f4f, master=self.f42)
            canvas.show()
            canvas.get_tk_widget().grid(row=0)
            canvas._tkcanvas.grid()
    
    def filtrUpdate(self,kk):     
        F = ['Regularized Least Squares', 'nu-method', 'Truncated SVD','Spectral cut-off','Iterative Landweber']
        print self.ftrO.get()
        if self.ftrO.get() == F[1] or self.ftrO.get() == F[4]:
            self.tMnO.state(['disabled'])
            self.nSvO.state(['disabled'])
            self.vScaleO.state(['disabled'])
        else:
            self.tMnO.state(['!disabled'])
            self.nSvO.state(['!disabled'])
            self.vScaleO.state(['!disabled'])

    def kpRangeUpdate(self,kk):
        if self.krnlO.get() =='Polynomial':
            self.kParaO.state(['disabled'])
            self.ukparO.state(['!disabled'])
            self.aSigmaO.state(['disabled'])
            self.ukparO.delete(0, 'end')
            self.ukparO.insert(0,1)
            
        elif self.krnlO.get() =='Gaussian':
            self.aSigmaO.state(['!disabled'])
            self.kParaO.state(['!disabled'])
            self.ukparO.state(['!disabled'])
            self.kParaO['from_']= 0.01
            self.kParaO['to']= 10.0
            #self.kParaO['increment'] =0.1
        else:
            self.kParaO.state(['disabled'])
            self.ukparO.state(['disabled'])
            self.aSigmaO.state(['disabled'])
            #self.kParaO['increment'] =1.0
        
        if self.krnlO.get() =='Linear':
            self.kParaO.state(['disabled'])
            self.ukparO.state(['disabled'])
            self.aSigmaO.state(['disabled'])

    def kParaSelFun(self,kk):
        if self.krnlO.get() =='Polynomial':
            dp = 0
        else:
            dp = 2
        
        kk = str(np.around(float(kk),dp))
        sr = "kpar : " + kk #str(0.23)
        self.label3f2.config(text=sr)
        self.ukparO.delete(0, 'end')
        self.ukparO.insert(0,np.around(float(kk),dp))
    
    def cKCVSelfun(self):
        if self.cKvcO.instate(['selected']):
            self.vScaleO.state(['!disabled'])
            self.splitO.state(['!disabled'])
            self.nSO.state(['!disabled'])
            self.tMnO.state(['!disabled'])
            self.tMxO.state(['!disabled'])
            self.nSvO.state(['!disabled'])
            self.fixVO.state(['disabled'])
        else:
            self.vScaleO.state(['disabled'])
            self.splitO.state(['disabled'])
            self.nSO.state(['disabled'])
            self.tMnO.state(['disabled'])
            self.tMxO.state(['disabled'])
            self.nSvO.state(['disabled'])
            self.fixVO.state(['!disabled'])
            
    def aSigmaSelFun(self):
        if self.aSigmaO.instate(['selected']):
            self.kParaO.state(['disabled'])
        else:
            self.kParaO.state(['!disabled'])
    
    def runLEARN(self):
        
        '''
        self.filter --->self.knl
        self.method --->self.filt
        '''
        # Final settings
        
        #Mappings------------------------------------------------
        
        taskMap ={1:'class',0:'regr'}
        
        filtMap = {'Regularized Least Squares':'rls', 'nu-method':'nu', 'Truncated SVD':'tsvd',
              'Spectral cut-off':'cutoff','Iterative Landweber':'land'}
        
        knlMap ={'Linear':'lin', 'Polynomial':'pol', 'Gaussian':'gauss'}
        
        splitMap ={'Sequential':'seq', 'Random':'rand'}
        
        #-----------------------------------------------------
         
        #input Data   #self.X, self.y, self.Xt, self.yt
        
        self.Task = taskMap[self.task.get()]
                                                                                        
        self.filt = filtMap[self.ftrO.get()]
        
        self.knl  = knlMap[self.krnlO.get()]
        
        '''
        if self.aSigmaO.instate(['selected']):
            self.kpar = rg.autosigma(self.X, 5)       
            sr = "kpar : " + str(np.around(self.kpar,4))
            self.label3f2.config(text=sr)
        '''

        if self.knl=='pol':
            #self.kpar = int(self.kParaO.get())
            self.kpar = int(self.ukparO.get())
            sr = "kpar : " + str(np.around(self.kpar,0))
            self.label3f2.config(text=sr)

        elif self.knl=='gauss':
            if self.aSigmaO.instate(['selected']):
                self.kpar = rg.autosigma(self.X, 5)       
            else:
                self.kpar = self.kParaO.get()
            
            if self.kpar==0:
                self.kpar=0.01

            sr = "kpar : " + str(np.around(self.kpar,4))
            self.label3f2.config(text=sr)
        else:
            self.kpar = 0
            sr = "kpar : " + 'NA'
            self.label3f2.config(text=sr)

            
        
        
        #KCV Split
        self.splitType = splitMap[self.splitO.get()]
        self.ksplit        = int(self.nSO.get())
        
        if self.fixedVO.instate(['selected']):
            self.trange   = float(self.fixVO.get())
        else:
            self.tmnx     = [float(self.tMnO.get()),float(self.tMxO.get())]
            self.nTvales  = int(self.nSvO.get())
            print self.tmnx, self.nTvales, type(self.nTvales)

            if self.filt in ['nu','land']:
                if float(self.tmnx[1]) ==int(self.tmnx[1]):
                    self.trange = int(self.tmnx[1])
                else:
                    tkMessageBox.Error('Tips and Tricks' ,'t max should be int')
            else:
                if self.vScaleO.get() == 'Linear Space':
                    self.trange = np.linspace(self.tmnx[0],self.tmnx[1],self.nTvales)
                else:
                    self.trange = np.logspace(self.tmnx[0],self.tmnx[1],self.nTvales)
        
        
        print "----------"
        print "Data       : ", self.Dtype
        print "ShapeData  : ", self.X.shape, self.y.shape, self.Xt.shape, self.yt.shape
        print "Task       : ", self.Task
        print "Fliter     : ", self.filt
        print "Kernal     : ", self.knl
        print "aSigma??   : ", self.aSigmaO.instate(['selected'])
        print "kpar       : ", self.kpar
        print "KCV??      : ", self.cKvcO.instate(['selected'])
        print "Split type : ", self.splitType
        print "#Splits    : ", self.ksplit
        print "trange     : ", self.trange
        print "----------"
        
               
        
        if self.cKvcO.instate(['selected']):
            
            #  kcv(knl, kpar, filt, t_range, X, y, k, task, split_type)
            t_kcv_idx, avg_err_kcv = rg.kcv(self.knl, self.kpar, self.filt, self.trange, self.X, self.y, 
                                       self.ksplit, self.Task, self.splitType)
           
            tval = 0;
            
            if self.filt in ['land','nu']:
                tval = t_kcv_idx + 1
                vmin = 0;
                vmax = np.max(self.trange)
            else:
                tval = self.trange[t_kcv_idx]
                vmin = np.min(self.trange)
                vmax = np.max(self.trange)
            
            #learn(knl, kpar, filt, t_range, X, y, task = False)
            alpha, err = rg.learn(self.knl, self.kpar, self.filt, tval, self.X, self.y, self.Task)
            print 'Trained....'
            index = np.argmin(err)
            #kernel(knl, kpar, X1, X2)
            K = rg.kernel(knl =self.knl , kpar = self.kpar, X1 =self.Xt, X2 =self.X)
            y_learnt = np.dot(K,alpha[index])
            lrn_error = rg.learn_error(y_learnt, self.yt, self.Task)
            
            self.TrainedAlpha = alpha[index]
            
            if np.isscalar(self.trange):
                self.trange = [self.trange]
                        
            
        
        else:
            
            self.trange
            #alpha, err = learn(handles.filter, kpar, handles.method, trange, X, y, handles.task)
            alpha, err = rg.learn(self.knl, self.kpar, self.filt, self.trange, self.X, self.y, self.Task)
            print 'Trained....'
            
            index = np.argmin(err)
            #kernel(knl, kpar, X1, X2)
            K = rg.kernel(knl =self.knl , kpar = self.kpar, X1 =self.Xt, X2 =self.X)
            y_learnt = np.dot(K,alpha[index])
            lrn_error = rg.learn_error(y_learnt, self.yt, self.Task)
            
            #K = rg.kernel(knl =self.knl , kpar = self.kpar, X1 =self.Xt, X2 =self.X)
            self.TrainedAlpha = alpha[index]
            
            
        #tt = np.random.rand()
        sr = 'Training Error ' + str(np.around(np.min(err),3))  
        self.label1f41.config(text=sr)
        tt = np.random.rand()
        sr = 'Testing Error  ' + str(np.around(lrn_error,3))
        self.label2f41.config(text=sr)
        tt = np.random.rand()
        if self.cKvcO.instate(['selected']):
            sr = 'Selected  t ' + str(np.around(tval,3))
        else:
            sr = 'Selected  t ' + str(np.around(self.trange,3))
        
        self.label3f41.config(text=sr)
        
        ##-------------Classifier boundries----------------
        nPoints = 100
        xx1 = [np.min(self.X[:,0]), np.min(self.Xt[:,0])] 
        yy1 = [np.max(self.X[:,0]), np.max(self.Xt[:,0])]
        xnn = np.min(xx1)
        xmx = np.max(yy1)
        X1mn = xnn
        X1mx = xmx
        ax = np.linspace(X1mn, X1mx, nPoints)
        
        xx1 = [np.min(self.X[:,1]), np.min(self.Xt[:,1])] 
        yy1 = [np.max(self.X[:,1]), np.max(self.Xt[:,1])]
        xnn = np.min(xx1)
        xmx = np.max(yy1)
        X2mn = xnn
        X2mx = xmx
        ay = np.linspace(X2mn, X2mx, nPoints)
        
        a, b = np.meshgrid(ax, ay)
        a = a.flatten('F')
        b = b.flatten('F')
        c = np.vstack([a,b]).T
        
        cK = rg.kernel(knl =self.knl , kpar = self.kpar, X1 = c, X2 =self.X)
        cyt = np.dot(cK,self.TrainedAlpha)
        
        self.c = c
        self.cyt =cyt
        self.Xmnmx =[X1mn,X1mx,X2mn,X2mx]
        
        self.trained = True
        self.plotTrain = True
        self.PlotTrTsTT()

        ##-------------Plot KCV..Error plot------------------------
        
        if self.cKvcO.instate(['selected']):
            self.avg_err_kcv = avg_err_kcv
            self.t_kcv_idx   = t_kcv_idx
            self.vnmx = [vmin,vmax]
            self.PlotKCVErro()

print "------------Ready----------------"


root = Tk()
App = RegML(root)
root.mainloop()