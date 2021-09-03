# -*- coding: utf-8 -*-
'''
Descrição das variáveis
    err: Vetor que conterá os melhores valores de fitness encontrados por execução.
    sr: Vetor que conterá valores binário: 1-Se atingiu o mínimo. 0-Não atingiu o mínimo. por execução.
    NCFO: Vetor que conterá o Número de Consultas a Função Objetivo por execução.
    time: Vetor que conterá os tempos de cada execução (em segundos).
    E: Número inteiro que contém quantas execuções o algorimto irá fazer.
    globalParams: Um dicionário que contém todos os paremtros necessários para execução das variantes do PSO.
    Chaves do dicionário 'globalParams':
        popSize: Deve conter um número inteiro que se refere ao valor de N (Quantas particulas terá no enxame)
        evalFunc: Deve conter o endereço de memória de uma função. Essa função deve receber como entrada um vetor X e retornar um valor escalar Y
        dm: Dever conter um número inteiro com a dimensionalidade do problema.
        stopC: Deve conter um valor que siginifca a diferença aceitavel como critério de parada entre os resultados da função utilizando a posição da memória global encontrada pelo algoritmo e a posição minima global conhecida. (-np.inf caso não queira utilizar o critério) 
        globalOpt: Deve conter o valor mínimo da função definida. (-np.inf caso não seja conhecido)
        iterationsMeta: Deve conter um úmero inteiro com o valor máximo T de iterações do algoritmo.
        NCFO: Número de Consultas a Função Objetivo, deve conter o valor máximo de consultas a função objetivo. (np.inf, caso não queira utilizar)
        print: Valor lógico utilizado para imprimir ou não a convergencia da solução por iteração.
    meta: Classe que contém o algoritmo a ser executado
    params: Dicionário que contém parametros especificos de cada algoritmo
    Chaves do dicionário 'params':
        c1
        c2
        c3
        iw
'''

import numpy as np
import timeit
import functions
from RISAPSO  import RISAPSO
from RISAXPSO import RISAXPSO
from XPSO     import XPSO
from RIARXPSO import RIARXPSO

err = []
sr = []
NCFO = []    
time = []    

E=5 

globalParams = {"popSize":20,
    "evalFunc":functions.f1,
    "dm":10,
    "bd":[-100,100],
    # "stopC":1e-8,
    "stopC":-np.inf,
    "globalOpt":0,    
    "iterationsMeta":6000,
    "NCFO":1e6,
    # "NCFO":np.inf,
    # "print":True
    "print":False
}

meta = RISAPSO
params = {
    "c1":1.4962,
    "c2":1e-2,        
    "c3":np.inf,        
    "iw":.7298,     
}

# meta = RISAXPSO
# params = {
#     "c1":[4   ,1   ,1.4962],
#     "c2":[1e-1,1e-5,1e-2],
#     "c3":[4   ,1   ,1.4962],          
#     "iw":[0.9, 0.4]
# }

# meta = XPSO
# params = {
#     "c1":[2,0,1.35],
#     "c2":[2,0,1.35],
#     "c3":[2,0,1.35],
#     "iw":[0.9, 0.4],     
# }

# meta = RIARXPSO
# params = {
#     "c1":[2,0,1.35],
#     "c2":[2,0,1.35],
#     "c3":[2,0,1.35],
#     "iw":[0.9, 0.4],     
# }

for e in range(E):  

    gp = {"popSize":globalParams["popSize"],
        "evalFunc":globalParams["evalFunc"],
        "dm":globalParams["dm"],
        "bd":globalParams["bd"],        
        "stopC":globalParams["stopC"],
        "globalOpt":globalParams["globalOpt"],
        "c1":params["c1"],
        "c2":params["c2"],        
        "c3":params["c3"],        
        "iw":params["iw"],     
        "iterationsMeta":globalParams["iterationsMeta"], 
        "NCFO":globalParams["NCFO"],
        "print":globalParams["print"],
    }

    exec_meta = meta(gp)

    starttime = timeit.default_timer()            
    exec_meta.execute()    
    timeExec = timeit.default_timer() - starttime    

    print("\nTerminou Execução",e+1)
    sr.append(exec_meta.getSr())
    err.append(exec_meta.getGlobalMin()-gp["globalOpt"])
    NCFO.append(exec_meta.getNCFO())
    time.append(timeExec)
    print('X',exec_meta.getGlobalMinX())        
    print('sr',sr[-1])
    print('err',err[-1])
    print('NCFO',NCFO[-1])    
    print('Time (s)',time[-1])    

    print("\nValores médios parciais:",meta)
    print('sr',np.mean(sr))
    print('err',np.mean(err))
    print('NCFO',np.mean(NCFO))
    print('time',np.mean(time))   

print("\nValores médios finais:",meta)
print('sr',np.mean(sr))
print('err',np.mean(err))
print('NCFO',np.mean(NCFO))
print('time',np.mean(time))
