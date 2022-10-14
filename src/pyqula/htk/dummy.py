
from .. import geometry
from .. import algebra
import numpy as np

def generate_dummy_hamiltonian(d,g=None):
    """Create a dummy Hamiltonian from a dictionary"""
    if g is None: g = geometry.chain() # create a chain
    h = g.get_hamiltonian() # create a dummy Hamiltonian
    h = h.get_multicell() # set as multicell Hamiltonian
    from ..multihopping import MultiHopping
    if type(d) is not dict: raise
    d = clean_dict(d) # overwrite
    h.set_multihopping(MultiHopping(d)) # set the dictionary
    h.geometry.supercell(h.intra.shape[0]) # dimensionality
    dim = get_dimensionality(d)
    h.geometry.dimensionality = dim
    h.geometry.a1 = np.array([1.,0.,0.])
    h.geometry.a2 = np.array([0.,1.,0.])
    h.geometry.a3 = np.array([0.,0.,1.])
    h.dimensionality = dim
    return h


def clean_dict(d):
    """Given a dictionary, return a cleaned up dictionary"""
    dout = dict()
    for key in d:
        key2 = tuple([int(k) for k in np.array(key)])
        dout[key2] = algebra.todense(np.array(d[key])) # to dense
    return dout


def get_dimensionality(d):
    """This function will try to guess the dimensionality of the system"""
    dim = 0 # zero dimensional
    vs = np.array([np.array(key) for key in d])
    from numpy.linalg import matrix_rank
    return matrix_rank(vs)
    


