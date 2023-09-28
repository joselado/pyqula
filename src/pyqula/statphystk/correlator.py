import numpy as np
from numba import jit

def get_nnc(g,den,n=20,normalized=False,**kwargs): 
    """Compute the first N correlators"""
    g0 = g.copy() ; g0.dimensionality = 0 # zero dimensional
    ds = g0.neighbor_distances() # neighbor distances
    if len(ds)<n: n = len(ds)
    r = g0.r # locations
    ri = r.copy() # copy locations
    deni = den.copy()
    if g.dimensionality>0: # periodic boundaries
        rj = g.multireplicas(1) # replicas of the locations
        den0 = den.copy() # copy
        denj = den0.copy() # start
        for i in range(1,len(rj)//len(den0)): denj = np.concatenate([denj,den0])
    else:
        rj = ri
        denj = deni
    cs = [get_nnci_jit(ri,rj,deni,denj,di,1e-3) for di in ds[0:n]]
    cs = np.array(cs) # convert to array
    if normalized: # normalize using Cauchy-Swartz inequality
        var = np.mean((den - np.mean(den))**2)
        # ignore replica effect due to periodic BC
        cs = cs/var # normalize by variance
    x,y = ds[0:len(cs)],cs # distance and correlators
    return x,y











@jit(nopython=True)
def get_nnci_jit_v1(ri,rj,deni,denj,di,delta):
    """Compute a single correlator"""
    out = 0. # output value
    no = 0. # counter
    outi,outj,noi,noj = 0.,0.,0.,0.
    for i1 in range(len(ri)):
        for i2 in range(len(rj)):
            r1 = ri[i1]
            r2 = rj[i2]
            dr = r1-r2
            dr2 = np.sum(dr*dr)
            if np.abs(dr2 - di**2)<delta:
                out += deni[i1]*denj[i2]
                no += 1.
                outi += deni[i1]
                outj += denj[i2]
                noi += 1
                noj += 1
    if no==0: return 0.
    return out/no  - outi/noi*outj/noj # return correlator








@jit(nopython=True)
def get_nnci_jit(ri,rj,deni,denj,di,delta):
    """Compute a single correlator"""
    out = 0. # output value
    no = 0. # counter
    deni = deni - np.mean(deni) # redefine so that it has zero average
    denj = denj - np.mean(denj) # redefine so that it has zero average
    for i1 in range(len(ri)):
        for i2 in range(len(rj)):
            r1 = ri[i1]
            r2 = rj[i2]
            dr = r1-r2
            dr2 = np.sum(dr*dr)
            if np.abs(dr2 - di**2)<delta:
                out += deni[i1]*denj[i2]
                no += 1.
    if no==0: return 0. # no neighbor found
    return out/no  #- np.mean(deni)*np.mean(denj) # return correlator
