# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g0 = geometry.honeycomb_lattice()
g = g0.get_supercell(2)
h = g.get_hamiltonian(has_spin=False)
ns = int(len(g.r)/len(g0.r)) # number of supercells
import numpy as np
from pyqula import unfolding
op = unfolding.bloch_projector(h,g0)
kpath = np.array(g.get_kpath(nk=200))*ns
op = None
h.get_fermi_surface(operator=op,nsuper=2,e=0.5)
#h.get_bands(operator=op,kpath=kpath)
from pyqula import kdos
#kdos.kdos_bands(h,operator=op,kpath=kpath)






