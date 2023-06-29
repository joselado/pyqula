# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
from pyqula import spectrum
g = geometry.triangular_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_onsite(-4.0)
h.add_pairing(delta=1.,mode="chiral_pwave")
fk = h.get_hk_gen()
def f(k):
    m = fk(k)
    return m[0,2]
from pyqula import spectrum
nk = 30 # number of kpoints
spectrum.reciprocal_map(h,lambda k: f(k),nk=nk,filename="DELTA.OUT") 
h.get_bands()
print(h.get_chern(nk=50))
print(h.get_gap())







