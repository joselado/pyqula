# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import groundstate
from pyqula import meanfield
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False) # create hamiltonian of the system
nk = 10
filling = 0.5

mf = meanfield.guess(h,"random") # initialization
scf = meanfield.Vinteraction(h,mf=mf,V2=2.0,nk=nk,filling=filling,mix=0.1)
print(scf.identify_symmetry_breaking())
h = h - scf.hamiltonian # get the Hamiltonian
#print("Topological invariant",h.get_topological_invariant())
h.get_bands() # calculate band structure
#from pyqula import topology
#groundstate.hopping(h)
#topology.write_berry(h)







