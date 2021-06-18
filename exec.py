# -*- coding: utf-8 -*-
import numpy as np
import timeit
import functions
from RISAPSO import RISAPSO

err = []
sr = []
NCFO = []    
time = []    

E=1
globalParams = {"popSize":20,
    "evalFunc":functions.rastrigin,
    "dm":2,
    "bd":[-100,100],
    # "stopC":1e-8,
    "stopC":-np.inf,
    "globalOpt":0.0,    
    "iterationsMeta":100,
    # "NCFO":1e6,
    "NCFO":np.inf,
    "print":True
    # "print":False
}

meta = RISAPSO
params = {
    "c1":1.4962,
    "c2":1e-2,        
    "c3":np.inf,        
    "iw":.7298,     
}


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

    print("\nMédias Parciais:")
    print('sr',np.mean(sr))
    print('err',np.mean(err))
    print('NCFO',np.mean(NCFO))
    print('time',np.mean(time))   

print("\nMédias Finais:",meta)
print('sr',np.mean(sr))
print('err',np.mean(err))
print('NCFO',np.mean(NCFO))
print('time',np.mean(time))