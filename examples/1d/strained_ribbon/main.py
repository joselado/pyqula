# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon(4) # create geometry of a zigzag ribbon

def tij(r1,r2): # function for strined hoppings
    dr = r1-r2 ; dr2 = dr.dot(dr)
    if abs(dr2-1.)<1e-3: # first neighbor
        return 1. + 1./(dr[1]**2+1.) # stronger hopping in the center
    return 0. # otherwise

h = g.get_hamiltonian(tij=tij) # create hamiltonian of the system
h.get_bands()







