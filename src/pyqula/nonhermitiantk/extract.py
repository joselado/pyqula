import numpy as np

def onsite(m,has_spin=True):
    """Extract the onsite energy"""
    if has_spin: # has spin degree of freedom
        n = m.shape[0]//2 # number of sites
        ds = np.zeros(n,dtype=np.complex_) # zeros
        for i in range(n):
          ds[i] = (m[2*i,2*i] + m[2*i+1,2*i+1])/2.
        return ds
    else:
        n = m.shape[0] # number of sites
        ds = np.zeros(n,dtype=np.complex_) # pairing
        for i in range(n):
          ds[i] = m[i,i]
        return ds


