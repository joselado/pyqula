# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import geometry
import numpy as np
from pyqula import embedding
from pyqula import parallel
# Here we will use the embedding method to calculate the
# density of states of a single vacancy in infinite graphene
# The embedding technique is a quite expensive algorithm, if you use
# large cells it will take a lot of time
g = geometry.honeycomb_lattice() # create geometry of a chain
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian,spinless
# create a new intraterm, vacancy is modeled as a large onsite potential
vintra = h.intra.copy() ; vintra[0,0] = 1000.0
parallel.cores = 4
energies = np.linspace(-1.5,1.5,200)
delta = 1e-2 # smearing
embedding.dos_impurity(h,vc=vintra,silent=False,energies=energies,
                      delta=delta,use_generator=False)
# results are written in DOS_DEFECTIVE.OUT and DOS_PRISTINE.OUT







