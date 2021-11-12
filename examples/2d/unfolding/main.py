# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g0 = geometry.honeycomb_lattice()
n  = 3
g = g0.get_supercell(n,store_primal=True)
h = g.get_hamiltonian(has_spin=False)
def ons(r):
  dr = r - g.r[0]
  if dr.dot(dr)<1e-1: return 100.0
  else: return 0.0

h.add_onsite(ons)

kpath = np.array(g.get_kpath(nk=200))*n
#op = None
h.get_fermi_surface(operator="unfold",nsuper=3,delta=3e-2,e=2.0,nk=80)
exit()
#h.get_bands(operator="unfold",kpath=kpath)
op = h.get_operator("unfold")#*h.get_operator("electron")
from pyqula import kdos
#kdos.kdos_bands(h,operator=op,kpath=kpath,delta=1e-1)
h.get_multi_fermi_surface(nk=50,energies=np.linspace(-4,4,100),
        delta=0.1,nsuper=n,operator="unfold")







