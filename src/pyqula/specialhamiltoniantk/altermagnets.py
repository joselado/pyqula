import numpy as np
from .. import geometry

def square_altermagnet(am=0.):
    g = geometry.square_lattice() # get the geometry
    return nwave_altermagnet(g,n=2,am=am)


def honeycomb_altermagnet(am=0.):
    raise
    g = geometry.honeycomb_lattice() # get the geometry
    return nwave_altermagnet(g,n=6,am=am)


def triangular_altermagnet(am=0.):
    raise
    g = geometry.triangular_lattice() # get the geometry
    return nwave_altermagnet(g,n=6,am=am)



def nwave_altermagnet(g,n=None,am=0.):
    """Create a Hamiltonian with n-wave altermagnetism"""
    if n is None: raise # this must be provided
    def ft(r1,r2):
        dr = r1-r2 # distance
        if 0.2<dr.dot(dr)<1.1: # nearest neighbor
            z = dr[0] + 1j*dr[1] # complex value
            zn = z**n # power
            tu = 1.+ am*z.real # up hopping
            td = 1.+ am*z.imag  # down hopping
            return np.array([[tu,0.],[0.,td]]) # hopping matrix
        else: return np.zeros((2,2))
    h = g.get_hamiltonian(tij=ft,is_multicell=False,
            has_spin=True,
            spinful_generator=True) # create the Hamiltonian
    h = h.get_multicell() # multicell Hamiltonian
    return h




