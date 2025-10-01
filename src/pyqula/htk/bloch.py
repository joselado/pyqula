import numpy as np
from numba import jit

from ..algebra import todense

def bloch_hamiltonian_generator_dense(h,hopping):
    """Return generator of a Bloch Hamiltonian"""
    if h.is_sparse:
        print("Only sparse Hamiltonians")
        raise
    ms,ds = [h.intra],[[0.,0.,0.]] # initialize
    for t in hopping: # loop over matrices
        ds.append(t.dir)
        ms.append(t.m)
    ms = [todense(m) for m in ms] # to dense, just in case
    return bloch_matrix_generator(ms,ds,dim=h.dimensionality)



def bloch_matrix_generator(ms,ds,dim=1):
    """Return a function that generates the Bloch Hamiltonian"""
    ms = np.array(ms,dtype=np.complex128)
    ds = np.array(ds,dtype=float)
    if dim==0: return lambda k: ms[0]
    ds = ds[:,0:dim] # crop to the dimension
    ds = np.ascontiguousarray(ds) # for memory efficiency
    ms = np.ascontiguousarray(ms) # for memory efficiency
    def f(k,**kwargs):
        return evaluate_bloch_matrix(ms,ds,np.array(k)[0:dim]) # call 
    return f


@jit(nopython=True)
def evaluate_bloch_matrix(ms,ds,k):
    """Evaluate the Bloch Hamiltonian"""
    out = ms[0]*0. # initialize
    n = len(ms) # number of matrices
    for i in range(n): # loop
        out += ms[i]*np.exp(1j*2*np.pi*ds[i].dot(k)) # Bloch phase
    return out
