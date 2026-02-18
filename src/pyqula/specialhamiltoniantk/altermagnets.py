import numpy as np
from .. import geometry

def square_altermagnet(am=0.):
    g = geometry.square_lattice() # get the geometry
    def ft(r1,r2):
        dr = r1-r2 # distance
        if 0.2<dr.dot(dr)<1.1: # nearest neighbor
            phi = np.angle(dr[0]+1j*dr[1]) # angle
            tu = 1.+am*np.cos(2*phi) # up hopping
            td = 1.-am*np.cos(2*phi)      # down hopping
            return np.array([[tu,0.],[0.,td]]) # hopping matrix
        else: return np.zeros((2,2))
    h = g.get_hamiltonian(tij=ft,is_multicell=False,
            has_spin=True,
            spinful_generator=True) # create the Hamiltonian
    h = h.get_multicell()
    return h


