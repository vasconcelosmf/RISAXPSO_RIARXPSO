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
        c1: [2.05,0,1.35] (List)
        c2: [2.05,0,1.35] (List)
        c3: [2.05,0,1.35] (List)
        iw: [0.9, 0.4]    (List)
'''
import numpy as np
import numpy.random as npr

class str_meta(type):
    def __str__(cls):
        return "XPSO"

class XPSO(metaclass = str_meta): 
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
        self.cc = [self.__parametros["c1"],self.__parametros["c2"],self.__parametros["c3"]]                                
        self.iw = np.linspace(self.__parametros["iw"][0], self.__parametros["iw"][1], num=self.T)

        # self.VMAX =  abs(self.bd[1] - self.bd[0]) * 0.2
        # self.VMAX =  abs(self.bd[1] - self.bd[0]) * 0.5
                           
        self.max_forget_ratio  = 0.6  
        self.min_forget_ratio  = 0.3 
        self.max_forget_ratio2 = 0.6 
        self.min_forget_ratio2 = 0.3 

        self.L = np.linalg.norm(np.ones((1,self.D)) * (abs(self.bd[1] - self.bd[0])))

        self.delta = 0.1        
        self.weight = 0.2
        self.cycle = 5
        self.elite_ratio=0.5
        self.factor = 0.01
        
        self.ps_elite=int(np.floor(self.N*self.elite_ratio))
        self.n_elite=int(np.floor(self.elite_ratio*self.N))         
    
    def execute(self):
        pop = npr.uniform(self.bd[0], self.bd[1], (self.N,self.D))
        fits = np.zeros(self.N)                       
        for i in range(self.N):
            fits[i] = self.__evalFunc(pop[i])
        
        self.setPop(pop,fits)

        self.neighbor = [[self.N-1,1]]
        for i in range(self.N-2):
            self.neighbor.append([i,i+2])
        self.neighbor.append([self.N-2,0])
        
        self.__globalMin = np.min(self.XFIT)        
        self.__globalMinId = np.argmin(self.XFIT)
        self.__globalPos = self.X[self.__globalMinId].copy()
        
        self.gbestrep = np.tile(self.__globalPos,(self.N,1))        
        self.gbest_stop = 0
        
        self.setDis()       

        self.forget_Num = np.zeros((self.N,1))
        self.forget_Dim = np.zeros((self.N,self.D))
        self.pmiu = np.zeros((self.N,1))
        self.setPmiu()
        self.pmiu = self.pmiu*self.forget_Dim
                       
        self.forget_Num2 = np.zeros((self.N,1))
        self.forget_Dim2 = np.zeros((self.N,self.D))
        self.pmiu2 = np.zeros((self.N,1))               
        self.setPmiu2()   
        self.pmiu2 = self.pmiu2*self.forget_Dim2        
        
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
        if self.gbest_stop >= self.cycle:
            self.cmiu1 = (1-self.weight) * self.cmiu1 + self.weight * np.tile(np.mean(self.good_cmiu1),(self.N,1))
            self.cmiu2 = (1-self.weight) * self.cmiu2 + self.weight * np.tile(np.mean(self.good_cmiu2),(self.N,1))
            self.cmiu3 = (1-self.weight) * self.cmiu3 + self.weight * np.tile(np.mean(self.good_cmiu3),(self.N,1))            
            self.c1 = self.randFCR(self.cmiu1, self.low1, self.up1)
            self.c2 = self.randFCR(self.cmiu2, self.low2, self.up2)
            self.c3 = self.randFCR(self.cmiu3, self.low3, self.up3)
                            
            self.setDis()

            self.forget_Num = np.zeros((self.N,1))
            self.forget_Dim = np.zeros((self.N,self.D))
            self.pmiu = np.zeros((self.N,1))
            self.setPmiu()
            self.pmiu = self.pmiu*self.forget_Dim
            
            self.forget_Num2 = np.zeros((self.N,1))
            self.forget_Dim2 = np.zeros((self.N,self.D))
            self.pmiu2 = np.zeros((self.N,1))               
            self.setPmiu2()                                  
            self.pmiu2 = self.pmiu2 * self.forget_Dim2
            
            # self.pbest_stop = np.zeros(self.N)
            # self.pbest_improve = np.zeros(self.N)
            self.gbest_stop = 0                                 

        self.gbest_changed = 0
                
        for j in range(self.N):
            self.tmpid = np.argmin(self.pbestval[self.neighbor[j]])                       
            self.getVelocity(j,i)    
            # self.truncVel(j)

            self.X[j] = self.X[j] + self.V[j]            
            self.truncSpace(j)
            
            self.XFIT[j] = self.__evalFunc(self.X[j])
            self.updateBest(j)       
        #FimForPop
           
        if self.gbest_changed == 1:
            self.gbest_stop = 0
        else:
            self.gbest_stop += 1
                
        IX = np.argsort(self.pbestval)
        selectedid = IX[:self.n_elite]
        
        self.good_cmiu1 = self.c1[selectedid]
        self.good_cmiu2 = self.c2[selectedid]
        self.good_cmiu3 = self.c3[selectedid]

        d = self.getDiversity()        
                                
        if self.__parametros["print"]:
            print("%s - %s" % (i+1,self.__globalMin))                                   
        self.__convergencia[i] += self.__globalMin
        self.__convDiv[i] += d
    
    def setPop(self,pop,fits):
        self.X = pop.copy()
        self.XFIT = fits.copy()        
        self.pbest = self.X.copy()
        self.pbestval = self.XFIT.copy()        
                
        self.aa = np.zeros((self.N,self.D))
        self.V = np.zeros((self.N,self.D))
        # self.V = npr.uniform(-self.VMAX, self.VMAX, (self.__parametros["popSize"],self.__parametros["dm"]))        
    
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
    
    def randFCR(self,cmiu, low, up):
        m,n = np.shape(cmiu)
        c = cmiu + self.delta * np.tan(np.pi * (npr.random((n,m)).T - 0.5))        
        c = np.clip(c,low,up)
        return c
    
    def setDis(self):
        self.dis = np.zeros(self.N)
        for i in range(self.N):
            self.dis[i] = np.linalg.norm(self.pbest[i]-self.gbestrep[i])                
        self.Index = np.argsort(self.dis)        
    
    def setPmiu(self):                  
        self.pmiu[self.Index[0:self.ps_elite-1]]=0       
        self.pmiu[self.Index[self.ps_elite:]]=np.log(np.arange(self.ps_elite+1,self.N+1).reshape((self.ps_elite, 1)))/(2*self.N)
        self.forget_Num[self.Index[self.ps_elite:]]=np.floor(self.D*(self.min_forget_ratio+(self.max_forget_ratio-self.min_forget_ratio)*\
            ((np.arange(1,self.N-self.ps_elite+1).reshape((self.N-self.ps_elite, 1)))/(self.N-self.ps_elite)))) 
        for k in range(self.N):                           
            Index1 = npr.randint(0,self.D,1)            
            if int(self.forget_Num[k]) == 1:
                self.forget_Dim[k][Index1[0]]=1                       
       
        self.pmiu = np.tile(self.pmiu, (1, self.D))
        self.radius = self.X.max(0) - self.X.min(0)
        self.radius = np.tile(self.radius, (self.N, 1))
        self.pmiu = self.pmiu * self.radius * self.factor
                       
    def setPmiu2(self):
        self.pmiu2[self.Index[0:self.ps_elite-1]]=0        
        self.pmiu2[self.Index[self.ps_elite:]]=np.log(np.arange(self.ps_elite+1,self.N+1).reshape((self.ps_elite, 1)))/(2*self.N)
        self.forget_Num2[self.Index[self.ps_elite:]]=np.floor(self.D*(self.min_forget_ratio2+(self.max_forget_ratio2-self.min_forget_ratio2)*\
            ((np.arange(1,self.N-self.ps_elite+1).reshape((self.N-self.ps_elite, 1)))/(self.N-self.ps_elite))))        
        
        for k in range(self.N):                        
            Index2 = npr.randint(0,self.D,1)
            if int(self.forget_Num2[k]) == 1:
                self.forget_Dim2[k][Index2[0]]=1
        
        self.pmiu2 = np.tile(self.pmiu2, (1, self.D))
        self.radius2 = self.X.max(0) - self.X.min(0)
        self.radius2 = np.tile(self.radius2, (self.N, 1))
        self.pmiu2 = self.pmiu2 * self.radius2 * self.factor

    def getVelocity(self,p,i):                
        self.aa[p] = self.c1[p] * npr.uniform(size=self.D) * (self.pbest[p]-self.X[p]) +\
                    self.c2[p] * npr.uniform(size=self.D) * (self.pbest[self.neighbor[p][self.tmpid]] * (1+self.pmiu2[p] * npr.randn(self.D)) - self.X[p]) +\
                    self.c3[p] * npr.uniform(size=self.D) * (self.gbestrep[p] * (1+self.pmiu[p] * npr.randn(self.D)) - self.X[p])
        self.V[p] = self.iw[i] * self.V[p] + self.aa[p]
                                
    def updateBest(self,p):      
        if self.XFIT[p] <= self.pbestval[p]:            
            self.pbest[p] = self.X[p].copy()
            self.pbestval[p] = self.XFIT[p].copy()              
            if self.pbestval[p] < self.__globalMin:
                self.__globalPos = self.pbest[p].copy()
                self.__globalMin = self.pbestval[p].copy()
                self.gbestrep = np.tile(self.__globalPos,(self.N,1))                
                self.gbest_changed = 1        
    
    def getDiversity(self):               
        avg = np.mean(self.X,axis=0)        
        d = np.sum(np.sqrt(np.sum((self.X - np.ones((self.N,self.D))*avg)**2,axis=1))) / (self.N * self.L)                                
        return d

    def truncVel(self,p):              
        self.V[p] = np.clip(self.V[p], -self.VMAX, self.VMAX)
    
    def truncSpace(self,p):        
        smin = self.X[p] < self.bd[0]
        smax = self.X[p] > self.bd[1]   

        # if True in smin or True in smax:                                                
        #     self.I[p] = 1            
        #     self.C[p] = 0           
        
        #Trunc
        self.X[p] = smax * self.bd[1] + (smax*-1+1) * self.X[p]
        self.X[p] = smin * self.bd[0] + (smin*-1+1) * self.X[p]
                        
        #ZeroV  
        self.V[p] = self.V[p] * (smin*-1+1)        
        self.V[p] = self.V[p] * (smax*-1+1)     

        # self.X[p] = np.clip(self.X[p], self.bd[0], self.bd[1])

        # if npr.uniform() < 0.5:
        #     self.X[p] = np.logical_and((self.X[p] <= np.tile(self.bd[1],self.D)),(self.X[p] >= np.tile(self.bd[0],self.D))) * self.X[p] +\
        #         (np.logical_or((self.X[p] > np.tile(self.bd[1],self.D)),(self.X[p] < np.tile(self.bd[0],self.D)))) *\
        #         (np.tile(self.bd[0],self.D) + (np.tile(self.bd[1],self.D) - np.tile(self.bd[0],self.D)) * npr.uniform(size=self.D))
        # else:                                                
        #     self.X[p] = np.logical_and((self.X[p] >= np.tile(self.bd[0],self.D)),(self.X[p] <= np.tile(self.bd[1],self.D))) * self.X[p] +\
        #         (self.X[p] < np.tile(self.bd[0],self.D)) * (np.tile(self.bd[0],self.D)+0.1 * (np.tile(self.bd[1],self.D)-np.tile(self.bd[0],self.D)) * npr.uniform(size=self.D)) +\
        #         (self.X[p] > np.tile(self.bd[1],self.D)) * (np.tile(self.bd[1],self.D)-0.1 * (np.tile(self.bd[1],self.D)-np.tile(self.bd[0],self.D)) * npr.uniform(size=self.D))
 
