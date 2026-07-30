# routines to extract the local Hamiltonian
import numpy as np
from ..algebra import todense


def site_dof(h):
    """Degrees of freedom per site: 1 (spinless), 2 (spinful or spinless
    Nambu), 4 (spinful Nambu) -- has_spin and has_eh each independently
    double the per-site block size."""
    dof = 2 if h.has_eh else 1
    if h.has_spin: dof *= 2
    return dof


def site_slice(h,i):
    """Slice of the matrix indices belonging to site i. Supports negative
    i (Python-style, counting from the last site), unlike a plain
    slice(i*dof,(i+1)*dof) would for i<0 -- the previous, pre-refactor
    version of this indexing relied on m[i,i] for the spinless case, which
    got negative indexing for free from numpy; a bare slice does not."""
    dof = site_dof(h)
    nsites = h.intra.shape[0]//dof
    if i<0: i += nsites
    if not (0<=i<nsites): raise IndexError("site index out of range: "+str(i))
    return slice(i*dof,(i+1)*dof)


def local_hamiltonian(h,m,i=0):
    """Given a certain Hamiltonian and a matrix,
    extract the local Hamiltonian"""
    m = todense(m) # dense array
    s = site_slice(h,i)
    if s.stop>m.shape[0]:
        print(i,m.shape[0])
        raise
    return np.array(m[s,s])


