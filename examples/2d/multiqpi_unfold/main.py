# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




import numpy as np
from pyqula import geometry
from pyqula import spectrum
g0 = geometry.square_lattice()
g0 = geometry.honeycomb_lattice()
ns = 2
g = g0.get_supercell(ns,store_primal=True)
h = g.get_hamiltonian(has_spin=False)
#h.add_sublattice_imbalance(.4)

def ons(r):
  dr = r - g.r[0]
  if dr.dot(dr)<1e-1: return 100.0
  else: return 0.0

h.add_onsite(ons)



from pyqula import unfolding
op = unfolding.bloch_projector(h,g0)
op = "unfold"
#op = None
h.get_qpi(delta=1e-2,mode="pm",operator=op,info=True,nsuper=2,nk=140,
  nunfold=ns)
#h.get_multi_fermi_surface(delta=1e-1,operator=op,nsuper=2,nk=30,
#energies = np.linspace(-2.0,2.0,300))







