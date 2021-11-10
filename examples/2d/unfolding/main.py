# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g0 = geometry.honeycomb_lattice()
n  = 2
g = g0.get_supercell(n,store_primal=True)
h = g.get_hamiltonian(has_spin=False)
ns = int(len(g.r)/len(g0.r)) # number of supercells
import numpy as np

from pyqula import potentials
v = potentials.commensurate_potential(g)
def ons(r):
  dr = r - g.r[0]
  if dr.dot(dr)<1e-1: return 1.0
  else: return 0.0

h.add_onsite(ons)
h.add_onsite(2.0)
h.add_swave(0.3)

kpath = np.array(g.get_kpath(nk=200))*n
#op = None
#h.get_fermi_surface(operator="unfold",nsuper=3,delta=3e-2,e=2.0,nk=100)
#h.get_bands(operator="unfold",kpath=kpath)
op = h.get_operator("unfold")*h.get_operator("electron")
from pyqula import kdos
kdos.kdos_bands(h,operator=op,kpath=kpath,delta=1e-1)
#h.get_multi_fermi_surface(nk=100,energies=np.linspace(-4,4,100),
#        delta=0.1,nsuper=3,operator="unfold")







