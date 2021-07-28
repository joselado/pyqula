# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import klist
from pyqula import specialgeometry
# get the geometry of ABC trilayer graphene
# distance between first neighbors is 1
# distance between layers is 3
g = specialgeometry.multilayer_graphene([-1,0,1])
#g = specialgeometry.multilayer_graphene([0])
g.write()
# define a function to compute the hopping in the multilayer
def fhop(ri,rj):
    """Function to compute the hopping"""
    dr = ri-rj ; dr2 = dr.dot(dr) # distance
    if 0.99<dr2<1.01: return 1.0 # first neighbors
    # interlayer hopping (distance between the layers is 3)
    if 8.99<dr2<9.01 and 2.99<abs(dr[2])<3.01: return 0.3 
    return 0.0 # else
g = g.supercell(13) # create a big supercell
# get the Hamiltonian
h = g.get_hamiltonian(fun=fhop,has_spin=False)
import numpy as np
from pyqula import potentials
f = potentials.commensurate_potential(g)
def fm(ri): # function for the mass
    if 2.9<ri[2]<3.1: # upper layer
        return f(ri)*0.2
    if -3.1<ri[2]<-2.9: # upper layer
        return f(ri)*0.2
    return 0.0 # otherwise
h.add_sublattice_imbalance(fm)
#h.shift_fermi(fm)
h.turn_sparse()
h.get_bands(num_bands=20)
#g = g.supercell(3)
#vs = [f(ri) for ri in g.r]
#np.savetxt("POTENTIAL.OUT",np.matrix([g.x,g.y,vs]).T)
#exit()
# compute the bandstructure
#h.get_bands()







