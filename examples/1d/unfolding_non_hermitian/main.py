# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
from pyqula import geometry
g0 = geometry.chain()
n  = 40
g = g0.get_supercell(n,store_primal=True)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True)
omega = 1./n
ons = lambda r: 0.1j*np.cos(np.pi*2*omega*r[0])
h.add_onsite(ons)

kpath = g.get_kpath()*n 

(ks,es,ds) = h.get_bands(operator="unfold",kpath=kpath)

import matplotlib.pyplot as plt

h.get_kdos_bands(operator="unfold",kpath=kpath,energies=np.linspace(0.,1.,200),
                eigmode="real")

exit()
plt.scatter(ks,es.real,s=ds.real)
#plt.scatter(ks,es.real)
plt.xlabel("Momentum") ; plt.xticks([]) ; plt.ylabel("Energy")
plt.show()
