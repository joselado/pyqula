# routines to extract the local Hamiltonian
import numpy as np
from ..algebra import todense



def local_hamiltonian(h,m,i=0):
    """Given a certain Hamiltonian and a matrix,
    extract the local Hamiltonian"""
    m = todense(m) # dense array
    if not h.has_spin and not h.has_eh: # spinless and no e-h
        if i>=m.shape[0]: 
            print(i,m.shape[0])
            raise
        return np.array([[m[i,i]]]) # return a single number
    elif h.has_spin and not h.has_eh: # spinful and no e-h
        if i>=m.shape[0]//2: raise
        return m[2*i:2*i+2,2*i:2*i+2] # return a 2x2 block
    elif h.has_spin and h.has_eh: # with e-h
        if i>=m.shape[0]//4: raise
        return m[4*i:4*i+4,4*i:4*i+4] # return a 4x4 block
    else: raise


