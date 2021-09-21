# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import geometry
import numpy as np
from pyqula import embedding
from pyqula import parallel
# Here we will use the embedding method to calculate the
# density of states of a defect in a nodal f-wave SC
# The embedding technique is a quite expensive algorithm, if you use
# large cells it will take a lot of time
g = geometry.triangular_lattice() # create geometry
h = g.get_hamiltonian() # get the Hamiltonian
h.add_pairing(mode="nodal_fwave",delta=0.1) # add nodal fwave
# create a Hamiltonian with a local impurity
r0 = g.r[0]
def V(r):
  dr = r-r0
  if dr.dot(dr)<0.2: return 1.0
  else: return 0.0
hv = h.copy()
hv.add_onsite(V) # add impurity

# now compute the pristine and defective DOS
parallel.cores = 6
energies = np.linspace(-0.7,0.7,100)
delta = 0.003 # smearing
embedding.dos_impurity(h,vc=hv.intra,silent=False,energies=energies,
                      delta=delta)
# results are written in DOS_DEFECTIVE.OUT and DOS_PRISTINE.OUT

