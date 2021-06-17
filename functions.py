import numpy as np

def rastrigin(sol):
    top = sol ** 2 - 10 * np.cos(2*np.pi*sol)        
    return 10 * len(sol) + top.sum()

def sphere(x):
    top = (x ** 2).sum()    
    return top

def rosenbrock(x):        
    top = 100 * ((x[1:] - x[:-1] ** 2) ** 2) + (1 - x[:-1]) ** 2    
    return top.sum()

def ackley(x):
    dim = len(x)  
    sum1 = (x ** 2).sum()
    sum2 = np.cos(2 * np.pi * x).sum()    
    return -20*np.exp(-0.2*np.sqrt(sum1/dim)) - np.exp(sum2/dim) + 20 + np.e

def griewank(x):
    dim = len(x)
    top = ((x ** 2) / 4000.0).sum()
    prod = np.prod(np.cos(x / np.sqrt(np.arange(1, dim+1))))    
    return top - prod + 1
