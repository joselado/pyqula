# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import geometry
from pyqula import topology

# build a topological superconductor
g = geometry.triangular_lattice()
h = g.get_hamiltonian()
h.add_rashba(.5) # Rashba spin-orbit coupling
h.add_zeeman([0.,0.,.5]) # Exchange field
h.add_onsite(-6.)
h.add_swave(0.1) 

nk = 20 # number of kpoints
# first compute the Chern number with a full mesh
c = h.get_chern(nk=nk)
print("Chern number in full BZ is ",c)

# now use a reduced mesh (50% of axis around the Gamma point)
from pyqula import klist
fraction = 0.5 # this is the fraction of each axis taken
kmesh = klist.partial_kmesh(g.dimensionality,nk=nk,f=fraction)
c = h.get_chern(kmesh=kmesh) # compute Chern number in a custom kmesh
print("Chern number in part of BZ is ",c)






