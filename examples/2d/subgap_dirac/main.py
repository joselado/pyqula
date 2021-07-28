# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import klist
import numpy as np
#g = geometry.single_square_lattice()
g = geometry.honeycomb_lattice()
#g = geometry.bichain()
#g = geometry.chain()
#g = g.supercell(3)
#g.dimensionality = 0
h = g.get_hamiltonian(has_spin=True)
#h.intra *= 0.0
h = h.get_multicell()
h.intra *= 1.5
#h.turn_sparse()
h.shift_fermi(0.8)
#h1 = h.copy()
#h.add_pairing(0.3,mode="swaveA")
h.add_pairing(0.4,mode="SnnAB")
h.add_pairing(0.4,mode="snn")
h.check()
#h.add_pairing(0.4,mode="snn")
#h.add_pairing(0.4,mode="swaveA")
#h.add_swave(0.4)
from pyqula import klist
kpath = klist.default(g,nk=4000)
h.get_bands(kpath=kpath)
exit()
#h.turn_sparse()
from pyqula import hamiltonians
#hamiltonians.print_hopping(h)
#h.check()
from pyqula import superconductivity
superconductivity.superconductivity_type(h)
exit()
#exit()
#h1.add_swave(0.3)
#print(h.same_hamiltonian(h1))
from pyqula import spectrum
spectrum.singlet_map(h,nsuper=1,mode="abs")
#h.get_bands()
#dos.dos(h,nk=100,use_kpm=True)







