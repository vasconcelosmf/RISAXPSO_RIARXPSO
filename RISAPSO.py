# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as npr
import copy

class str_meta(type):
    def __str__(cls):
        return "RI-SAPSO"

class RISAPSO(metaclass = str_meta):

    def evalFuncCount(self,x):   
        self.__NCFO += 1        
        return self.__parametros["evalFunc"](x)

    def __init__(self, parametros):
        self.__parametros = parametros
        self.X = []
        self.__globalMin = np.inf
        self.__globalPos = np.zeros(parametros["dm"])        
        # self.__evalFunc = parametros["evalFunc"]
        self.__evalFunc = self.evalFuncCount        
        self.__sr = 0
        self.__NCFO = 0
        self.__STOPC = parametros["stopC"]
        self.__g_opt = parametros["globalOpt"]

        self.D = parametros["dm"]
        self.bd = self.__parametros["bd"]
        self.N = self.__parametros["popSize"]
        self.T = self.__parametros["iterationsMeta"]
        self.__convergencia = np.zeros(self.T)
        self.__convDiv = np.zeros(self.T)        
        self.k = 1
        self.CMAX = 3
        self.dir = 1 #attractive as default        
        self.alpha = 3
        self.DT = [1e-2, .25]
        self.params = {"IW":self.__parametros["iw"],            
            "SC":self.__parametros["c1"], #c1
            "GC":self.__parametros["c2"], #c2
            "d":self.D,
            "a":self.alpha*np.pi/180,
            "i":np.eye(self.D)
            }
        self.L = np.linalg.norm(np.ones((1,self.D)) * (abs(self.bd[1] - self.bd[0])))        
        self.VMAX = self.k * abs(self.bd[1] - self.bd[0])/2        
                   
    def execute(self):
        pop = npr.uniform(self.bd[0], self.bd[1], (self.N,self.D))
        fits = np.zeros(self.N)                       
        for i in range(self.N):
            fits[i] = self.__evalFunc(pop[i])

        self.setPop(pop,fits)

        self.__globalMin = np.min(self.XFIT)
        self.__globalPos = self.X[np.argmin(self.XFIT)].copy()
                        
        if self.__parametros["print"]:
            print("%s - %s" % (1,self.__globalMin))            
        self.__convergencia[0] += self.__globalMin
        self.__convDiv[0] += self.getDiversity()
        
        # self.iterate(1)
        # Inicia as iterações                                   
        for i in range(1,self.T,1):                
            self.iterate(i)
            if (self.__globalMin - self.__g_opt) <= self.__STOPC:
                self.__sr = 1                
                self.__convergencia[i+1:] += self.__globalMin
                self.__convDiv[i+1:] += self.__convDiv[-1]
                break   
            if (self.__NCFO >= self.__parametros["NCFO"]):
                if i+1 < self.T:
                    self.__convergencia[i+1:] += self.__globalMin    
                    self.__convDiv[i+1:] += self.__convDiv[-1]                
                break                        
        
    def iterate(self,i):
        for j in range(self.N):                        
            if self.I[j] == 0:                    
                self.getGradient(j)
                self.truncGrad(j)          
            
            self.getVelocity(j)
            self.truncVel(j)
            
            self.X[j] = self.X[j] + self.V[j]                
            self.truncSpace(j)
            
            self.XFIT[j] = self.__evalFunc(self.X[j])
            self.updateBest(j)            
        
        self.updateImportance()
        self.OLDXFIT = copy.deepcopy(self.XFIT)
        d = self.getDiversity()
        self.updateDir(d)
        
        if self.__parametros["print"]:
            print("%s - %s" % (i+1,self.__globalMin))
        self.__convergencia[i] += self.__globalMin
        self.__convDiv[i] += d
                
    def setPop(self,pop,fits):        
        self.I = np.zeros((self.__parametros["popSize"],1))        
        self.C = np.zeros((self.__parametros["popSize"],1))
        self.G = np.zeros((self.__parametros["popSize"],self.__parametros["dm"]))        
        self.X = pop.copy()
        self.XFIT = fits.copy()
        self.V = npr.uniform(-self.VMAX, self.VMAX, (self.__parametros["popSize"],self.__parametros["dm"]))                        
        self.P = pop.copy()
        self.PFIT = fits.copy()
        self.OLDXFIT = fits.copy()        
    
    def getGlobalMin(self):
        return self.__globalMin         
    
    def getSr(self):
        return self.__sr
    
    def getNCFO(self):
        return self.__NCFO
    
    def setNCFO(self,NCFO):
        self.__NCFO = NCFO
    
    def getGlobalMinX(self):
        return self.__globalPos
    
    def getConv(self):
        return self.__convergencia

    def getConvDiv(self):
        return self.__convDiv
                          
    def getGradient(self,p):     
        self.G[p] = np.zeros(self.D)
        fx = self.XFIT[p].copy()                
        step = 1e-5        
        for i in range(self.D):
            xli = self.X[p].copy()            
            xli[i] = self.X[p][i] + step
            self.G[p][i] = ( self.__evalFunc(xli) - fx ) / step
                    
    def truncGrad(self,p):
        self.G[p] = np.clip(self.G[p], -self.VMAX, self.VMAX)        
    
    def m_function(self):        
        rand = npr.random((self.D,self.D))    
        A = -.5 + rand                        
        M = self.params["i"] + self.params["a"] * (A - A.conj().T)
        return M

    def getVelocity(self,p):        
        if self.dir == 1:                        
            self.V[p] = self.params["IW"] * self.V[p] + self.dir *(\
                self.I[p] * self.params["SC"] * npr.uniform() * (self.__globalPos - self.X[p]) +\
                (self.I[p] - 1) * self.params["GC"] * npr.uniform() * self.G[p])
        else:            
            M1 = self.m_function()                        
            M2 = self.m_function()
            self.V[p] = self.params["IW"] * self.V[p] + self.dir *(\
                self.I[p] * self.params["SC"] * npr.uniform() * (M1 @ (self.__globalPos - self.X[p]).conj().T).conj().T +\
                (self.I[p] - 1) * self.params["GC"] * npr.uniform() * (M2 @ self.G[p].conj().T).conj().T)              
        
    def truncVel(self,p):              
        self.V[p] = np.clip(self.V[p], -self.VMAX, self.VMAX) 
    
    def truncSpace(self,p):
        if len(np.argwhere(self.X[p] < self.bd[0])) >= 1 or len(np.argwhere(self.X[p] > self.bd[1])) >= 1:                                                
            self.I[p] = 1            
            self.C[p] = 0
        self.X[p] = np.clip(self.X[p], self.bd[0], self.bd[1])
    
    def updateBest(self,p):        
        if self.XFIT[p] < self.PFIT[p]:
            self.P[p] = self.X[p].copy()
            self.PFIT[p] = self.XFIT[p].copy()
            if self.XFIT[p] < self.__globalMin:                
                self.__globalMin = self.XFIT[p].copy()
                self.__globalPos = self.X[p].copy()
    
    def updateImportance(self):        
        for i in range(self.N): 
            if self.I[i] == 0: # gradient                                                   
                if abs(self.XFIT[i] - self.OLDXFIT[i]) < 1e-2 or np.linalg.norm(self.G[i]) < 1e-2:
                    self.C[i] += 1                    
                    if self.C[i] == self.CMAX:                             
                        self.I[i] = 1
                        self.C[i] = 0
                else:
                    self.C[i] = 0                
            elif np.sqrt(np.sum((self.X[i] - self.__globalPos)**2,axis=0)) < 1e-5:  # global                                                       
                self.I[i] = 0
                self.C[i] = 0    
    
    def getDiversity(self):               
        avg = np.mean(self.X,axis=0)        
        d = np.sum(np.sqrt(np.sum((self.X - np.ones((self.N,self.D))*avg)**2,axis=1))) / (self.N * self.L)                                
        return d
    
    def updateDir(self,d):              
        if self.dir > 0 and d < self.DT[0]: # must repulse            
            self.dir = -1                           
            self.I = np.ones((self.N,1))                             
        elif self.dir < 0 and d > self.DT[1]: # must attract
            self.dir = 1
            self.I = np.zeros((self.N,1))                                                      