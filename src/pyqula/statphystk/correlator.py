import numpy as np
from numba import jit

def get_nnc_v0(g,den,rmax=10.0,dr=1.0): 
    """Compute the first N correlators"""
    ds = np.linspace(1.,rmax,int(rmax/dr)) # output
    r = g.r # locations
    if g.dimensionality>0: # periodic boundaries
        r = g.multireplicas(1) # replicas of the locations
        den0 = den.copy() # copy
        den = den0.copy() # start
        for i in range(1,len(r)//len(ds0)): den = np.concatenate([den,den0])
    cs = [get_nnci_jit(r,den,di,dr/2.) for di in ds]
    return ds,cs



def get_nnc(g,den,n=20,**kwargs): 
    """Compute the first N correlators"""
    g0 = g.copy() ; g0.dimensionality = 0 # zero dimensional
    ds = g0.neighbor_distances() # neighbor distances
    if len(ds)<n: n = len(ds)
    r = g0.r # locations
    cs = [get_nnci_jit(r,den,di,1e-3) for di in ds[0:n]]
    x,y = ds[0:len(cs)],np.array(cs) # distance and correlators
    return x,y



















@jit(nopython=True)
def get_nnci_jit(r,den,di,delta):
    """Compute a single correlator"""
    out = 0. # output value
    no = 0 # counter
    for i1 in range(len(r)):
        for i2 in range(len(r)):
            r1 = r[i1]
            r2 = r[i2]
            dr = r1-r2
            dr2 = np.sum(dr*dr)
            if np.abs(dr2 - di**2)<delta:
                out += den[i1]*den[i2]
                no += 1
    if no==0: return 0.
    return out/no - np.mean(den)**2 # return correlator
