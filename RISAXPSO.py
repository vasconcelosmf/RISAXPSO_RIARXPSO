# -*- coding: utf-8 -*-
'''
Chaves do dicionário 'parametros':
        popSize: Deve conter um número inteiro que se refere ao valor de N (Quantas particulas terá no enxame)
        evalFunc: Deve conter o endereço de memória de uma função. Essa função deve receber como entrada um vetor X e retornar um valor escalar Y
        dm: Dever conter um número inteiro com a dimensionalidade do problema.
        stopC: Deve conter um valor que siginifca a diferença aceitavel como critério de parada entre os resultados da função utilizando a posição da memória global encontrada pelo algoritmo e a posição minima global conhecida. (-np.inf caso não queira utilizar o critério) 
        globalOpt: Deve conter o valor mínimo da função definida. (-np.inf caso não seja conhecido)
        iterationsMeta: Deve conter um úmero inteiro com o valor máximo T de iterações do algoritmo.
        NCFO: Número de Consultas a Função Objetivo, deve conter o valor máximo de consultas a função objetivo. (np.inf, caso não queira utilizar)
        print: Valor lógico utilizado para imprimir ou não a convergencia da solução por iteração.
        c1: [4 ,1 ,1.4962]   (List)
        c2: [1e-1,1e-5,1e-2] (List)
        c3: [4 ,1 ,1.4962]   (List)
        iw: [0.9, 0.4]       (List)        
'''

import numpy as np
import numpy.random as npr
import copy

class str_meta(type):
    def __str__(cls):
        return "RI-SAXPSO"

class RISAXPSO(metaclass = str_meta):                        

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
        self.CMAX = 3
        self.alpha = 3
        
        self.DT = np.logspace(1, 10, num=self.T,base=0.25)  # y = base^x                
        
        self.delta = 0.1
        self.weight = 0.2
        self.cycle = 5
        self.elite_ratio = 0.5

        self.cc = [self.__parametros["c1"],self.__parametros["c2"],self.__parametros["c3"]]
        self.iw = np.linspace(self.__parametros["iw"][0], self.__parametros["iw"][1], num=self.T)

        self.params = {
            "CC":self.__parametros["c3"], #c3
            "SC":self.__parametros["c1"], #c1
            "GC":self.__parametros["c2"], #c2
            "d":self.D,
            "a":self.alpha*np.pi/180,
            "i":np.eye(self.D)
            }
        self.L = np.linalg.norm(np.ones((1,self.D)) * (abs(self.bd[1] - self.bd[0])))        
                
        # self.VMAX =  abs(self.bd[1] - self.bd[0]) * 0.2
        # self.VMAX =  abs(self.bd[1] - self.bd[0]) * 0.5
                   
    def execute(self):
        pop = npr.uniform(self.bd[0], self.bd[1], (self.N,self.D))
        fits = np.zeros(self.N)                       
        for i in range(self.N):
            fits[i] = self.__evalFunc(pop[i])

        self.setPop(pop,fits)
        self.__globalMin = np.min(self.XFIT)
        self.__globalPos = self.X[np.argmin(self.XFIT)].copy()
                
        self.gbest_stop = 0
        
        self.cmiu1 = self.cc[0][2]*np.ones((self.N,1))
        self.low1 = self.cc[0][1]
        self.up1 = self.cc[0][0]

        self.cmiu2 = self.cc[1][2]*np.ones((self.N,1))
        self.low2 = self.cc[1][1]
        self.up2 = self.cc[1][0]

        self.cmiu3 = self.cc[2][2]*np.ones((self.N,1))
        self.low3 = self.cc[2][1]
        self.up3 = self.cc[2][0]
        
        self.c1 = self.randFCR(self.cmiu1, self.low1, self.up1)
        self.c2 = self.randFCR(self.cmiu2, self.low2, self.up2)    
        self.c3 = self.randFCR(self.cmiu3, self.low3, self.up3)    
        
        self.n_elite=int(np.floor(self.elite_ratio*self.N)) 

        self.good_cmiu1 = 0
        self.good_cmiu2 = 0        
        self.good_cmiu3 = 0        
                                
        if self.__parametros["print"]:
            print("%s - %s" % (1,self.__globalMin))  
        self.__convergencia[0] += self.__globalMin
        self.__convDiv[0] += self.getDiversity()
                
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
        self.updateDir(self.__convDiv[i-1],i)
        
        self.gbest_changed = 0    

        for j in range(self.N):                        
            if self.I[j] == 0:                    
                self.getGradient(j)
                # self.truncGrad(j)          
            
            self.getVelocity(j,i)            
            # self.truncVel(j)
                        
            self.X[j] = self.X[j] + self.V[j]            
            self.truncSpace(j)
                        
            self.XFIT[j] = self.__evalFunc(self.X[j])
            self.updateBest(j)   

        if self.gbest_changed == 1:
            self.gbest_stop = 0
        else:
            self.gbest_stop += 1         

        IX = np.argsort(self.PFIT)
        selectedid = IX[:self.n_elite]
        
        self.good_cmiu1 = self.c1[selectedid]
        self.good_cmiu2 = self.c2[selectedid]
        self.good_cmiu3 = self.c3[selectedid]
        
        self.updateImportance()
        self.OLDXFIT = copy.deepcopy(self.XFIT)        
        
        if self.__parametros["print"]:
            print("%s - %s" % (i+1,self.__globalMin))            
        self.__convergencia[i] += self.__globalMin
        self.__convDiv[i] += self.getDiversity()
            
    def setPop(self,pop,fits):        
        self.I = np.zeros((self.N,1))        
        self.C = np.zeros((self.N,1))
        self.G = np.zeros((self.N,self.D))        
        self.X = pop.copy()
        self.XFIT = fits.copy()
        self.V = np.zeros((self.N,self.D))        
        # self.V = npr.uniform(-self.VMAX, self.VMAX, (self.N,self.D))                        
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

    def getVelocity(self,p,i):
        if self.dir == 1:            
            aa = self.I[p] * self.c3[p] * npr.uniform() * (self.P[p] - self.X[p]) +\
                 self.I[p] * self.c1[p] * npr.uniform() * (self.__globalPos - self.X[p]) +\
           (self.I[p] - 1) * self.c2[p] * npr.uniform() * self.G[p]
        else:
            M1 = self.m_function()
            M2 = self.m_function()
            aa = self.I[p]  * self.c1[p] * npr.uniform() * (M1 @ (self.__globalPos - self.X[p]).conj().T).conj().T +\
            (self.I[p] - 1) * self.c2[p] * npr.uniform() * (M2 @ self.G[p].conj().T).conj().T                                                
            aa = -1 * aa      
                        
        self.V[p] = self.iw[i] * self.V[p] + aa
        
    def truncVel(self,p):              
        self.V[p] = np.clip(self.V[p], -self.VMAX, self.VMAX) 
            
    def truncSpace(self,p):        
        smin = self.X[p] < self.bd[0]
        smax = self.X[p] > self.bd[1]   

        if True in smin or True in smax:                                                
            self.I[p] = 1            
            self.C[p] = 0           
        
        #Trunc
        self.X[p] = smax * self.bd[1] + (smax*-1+1) * self.X[p]
        self.X[p] = smin * self.bd[0] + (smin*-1+1) * self.X[p]
                        
        #ZeroV  
        self.V[p] = self.V[p] * (smin*-1+1)        
        self.V[p] = self.V[p] * (smax*-1+1)     

        # self.X[p] = np.clip(self.X[p], self.bd[0], self.bd[1])
    
    def updateBest(self,p):        
        if self.XFIT[p] < self.PFIT[p]:            
            self.P[p] = self.X[p].copy()
            self.PFIT[p] = self.XFIT[p].copy()
            if self.XFIT[p] < self.__globalMin:                
                self.__globalMin = self.XFIT[p].copy()
                self.__globalPos = self.X[p].copy()
                self.gbest_changed = 1
    
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
    
    def updateDir(self,d,i): 
        stag = self.gbest_stop >= self.cycle
        if stag:             
            self.updateParams()                                    
            self.gbest_stop = 0        
        elif d < self.DT[i]: # must repulse
            self.dir = -1
            self.I = np.ones((self.N,1))                                         
        else:    # must attract
            self.dir = 1          

    def updateParams(self):
        self.cmiu1 = (1-self.weight) * self.cmiu1 + self.weight * np.tile(np.mean(self.good_cmiu1),(self.N,1))
        self.cmiu2 = (1-self.weight) * self.cmiu2 + self.weight * np.tile(np.mean(self.good_cmiu2),(self.N,1))                
        self.cmiu3 = (1-self.weight) * self.cmiu3 + self.weight * np.tile(np.mean(self.good_cmiu3),(self.N,1))                
        self.c1 = self.randFCR(self.cmiu1, self.low1, self.up1)
        self.c2 = self.randFCR(self.cmiu2, self.low2, self.up2)
        self.c3 = self.randFCR(self.cmiu3, self.low3, self.up3)
    
    def randFCR(self,cmiu, low, up):
        m,n = np.shape(cmiu)                        
        c = np.clip(cmiu + self.delta * np.tan(np.pi * (npr.random((n,m)).T - 0.5)),low,up)        
        return c
