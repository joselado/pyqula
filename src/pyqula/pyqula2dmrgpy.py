# routines to transform a pyqula Hamiltonian into a dmrgpy one
from . import algebra
import numpy as np

def generate_fermionic_chain(H):
    """Return a Fermionic object from dmrgpy to solve the Hamiltonian
    exactly"""
    from dmrgpy import fermionchain
    if H.dimensionality!=0:
        print("Only implemented for 0d dimensional Hamiltonians")
    n = H.intra.shape[0] # number of total orbitals
    if H.has_spin:
        fc = fermionchain.Spinful_Fermionic_Chain(n//2) # generate chain
    else:
        fc = fermionchain.Fermionic_Chain(n) # generate chain
    hp = 0. # initialize
    m = algebra.todense(H.intra) # get the dense matrix
    for i in range(n):
        for j in range(n):
            tij = m[i,j] # hopping
            if np.abs(tij)>1e-5:
                hp = hp + tij*fc.Cdag[i]*fc.C[j]
    fc.set_hamiltonian(hp) # set up the Hamiltonian
    return fc # return the fermionic chain object


