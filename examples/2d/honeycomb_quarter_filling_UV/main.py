# Add the root path of the pyqula library
import os ; import sys 
#sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



# zigzag ribbon
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian() # create hamiltonian of the system
# create a dummy Hamiltonian to use as initial guess
h0 = h.copy() # copy
h0.add_sublattice_imbalance(0.2) # add CDW
h0.add_zeeman(2.0) # add magnetic order

# now perform the SCF calculation
h = h.get_mean_field_hamiltonian(U=6.0,V1=1.5,verbose=1,
              mf=h0,mix=0.9,filling = 0.25)
# compute bands
h.get_bands(operator="sz")







