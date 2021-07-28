import numpy as np
from numba import jit

def Vijkl(h,n=4,fv=None):
    """Return the Coulomb matrix elements between different orbitals"""
    numw = n
    if h.dimensionality!=0: raise # only for 0D
    if h.has_eh: raise # not implemented
    if h.has_spin: raise # not implemented
    es,ws = h.get_eigenvectors(numw=numw) # get eigenstates
    g = h.geometry # geometry
    if fv is None:
        def fv(r):
            r2 = r.dot(r)
            if r2<0.1: return 0.0 # zero
            else: return 1./np.sqrt(r2) # coulomb
    vs = [[fv(r1-r2) for r1 in g.r] for r2 in g.r] # interaction
    out = np.zeros((numw,numw,numw,numw),dtype=np.complex) # storage
    return get_vijkl_jit(ws,np.array(vs),out)


@jit(nopython=True)
def get_vijkl_jit(ws,vs,out):
    """Return the Vijkl elements, given a distance interaction vs"""
    n = len(ws) # number of wavefunctions
    nc = len(vs) # number of components
    for i in range(n):
        wi = np.conjugate(ws[i])
        for j in range(n):
            wj = ws[j]
            for k in range(n):
                wk = np.conjugate(ws[k])
                for l in range(n):
                    wl = ws[l]
                    o = 0.0j # initialize
                    for ii in range(nc):
                        for jj in range(nc):
                            o = o + vs[ii,jj]*wi[ii]*wj[ii]*wk[jj]*wl[jj]
                    out[i,j,k,l] = o # store
    return out # return output
