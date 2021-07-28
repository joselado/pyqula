import numpy as np
from ..superconductivity import time_reversal

def has_time_reversal_symmetry(h):
    """Check if a Hamiltonian breaks time reversal symmetry"""
    if h.has_eh: 
        print("WARNING, time reversal not implemented for BdG")
        return False
    else:
        if h.has_spin: f = time_reversal
        else: f = lambda x: np.conjugate(x)
        h1 = h.copy()
        h1.modify_hamiltonian_matrices(f) # apply time reversal
        d0 = h.get_multihopping()
        d1 = h1.get_multihopping()
        dd = d0 -d1 # difference
        diff = dd.dot(dd) # scalar product
        if diff.real<1e-7: return True
        else: return False


