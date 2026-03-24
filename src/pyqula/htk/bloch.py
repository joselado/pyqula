import numpy as np
from numba import jit

from ..algebra import todense

def bloch_hamiltonian_generator_dense(h,hopping,**kwargs):
    """Return generator of a Bloch Hamiltonian"""
    if h.is_sparse:
        print("Only sparse Hamiltonians")
        raise
    ms,ds = [h.intra],[[0.,0.,0.]] # initialize
    for t in hopping: # loop over matrices
        ds.append(t.dir)
        ms.append(t.m)
    ms = [todense(m) for m in ms] # to dense, just in case
    return bloch_matrix_generator(ms,ds,dim=h.dimensionality,**kwargs)



def bloch_matrix_generator(ms,ds,dim=1,use_jax=False):
    """Return a function that generates the Bloch Hamiltonian"""
    ms = np.array(ms,dtype=np.complex128)
    ds = np.array(ds,dtype=np.float64)
    if dim==0: return lambda k: ms[0]
    ds = ds[:,0:dim] # crop to the dimension
    ds = np.ascontiguousarray(ds) # for memory efficiency
    ms = np.ascontiguousarray(ms) # for memory efficiency
    if use_jax:
        def f(k,**kwargs):
            return evaluate_bloch_matrix(ms,ds,jnp.array(k)[0:dim]) # call 
        return f
    else:
        def f(k,**kwargs):
            return evaluate_bloch_matrix_jit(ms,ds,np.array(k)[0:dim]) # call 
        return f


@jit(nopython=True)
def evaluate_bloch_matrix_jit(ms,ds,k):
    """Evaluate the Bloch Hamiltonian"""
    out = ms[0]*0. # initialize
    n = len(ms) # number of matrices
    for i in range(n): # loop
        out += ms[i]*np.exp(1j*2*np.pi*ds[i].dot(k)) # Bloch phase
    return out


import jax
import jax.numpy as jnp

def evaluate_bloch_matrix_jax(ms, ds, k):
    ms_arr = jnp.stack(ms)          
    ds_arr = jnp.stack(ds)          
    dk = jnp.dot(ds_arr, k)         
    phases = jnp.exp(1j * 2 * jnp.pi * dk)   
    out = jnp.einsum('nij,n->ij', ms_arr, phases)
    return out

#def evaluate_bloch_matrix_jax(ms,ds,k):
#    """Evaluate the Bloch Hamiltonian"""
#    out = ms[0]*0. # initialize
#    n = len(ms) # number of matrices
#    for i in range(n): # loop
#        dk = jnp.sum(ds[i]*k) # dot product
#        out += ms[i]*jnp.exp(1j*2*jnp.pi*dk) # Bloch phase
#    return out

evaluate_bloch_matrix = jax.jit(evaluate_bloch_matrix_jax)

