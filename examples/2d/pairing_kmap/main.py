# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
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
(ks,bs) = spectrum.reciprocal_map(h,lambda k: f(k),nk=nk,filename="DELTA.OUT")
(k,e) = h.get_bands()
print(h.get_chern(nk=50))
print(h.get_gap())

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(k,e)
plt.xlabel("k-path")
plt.ylabel("Energy")

plt.figure()
plt.scatter(ks[:,0],ks[:,1],c=np.abs(bs))
plt.colorbar(label="|$\\Delta$|")
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
plt.show()







