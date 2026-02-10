# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
from pyqula import geometry
g0 = geometry.chain()
n  = 10
g = g0.get_supercell(n,store_primal=True)
h = g.get_hamiltonian(has_spin=False)

from pyqula import potentials
v = potentials.commensurate_potential(g,k=23)
def ons(r):
  dr = r - g.r[0]
  if dr.dot(dr)<1e-1: return 2.0
  else: return 0.0

#h.geometry.write_profile(v,nrep=1)
h.add_onsite(ons)
#h.add_onsite(2.0)
#h.add_swave(0.3)

from pyqula import kdos
kpath = np.array(g.get_kpath(nk=200))*n
op = h.get_operator("unfold")#*h.get_operator("electron")
kdos.kdos_bands(h,operator=op,kpath=kpath,delta=1e-1)




