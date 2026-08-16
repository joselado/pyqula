# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Li-Haldane entanglement spectrum of a Chern insulator
# The Haldane model is cut into a ring of unit cells along a2, keeping the
# momentum along a1 as a good quantum number. Half of the ring is traced
# out, and the eigenvalues xi_n of the single-particle entanglement
# Hamiltonian are plotted against that momentum. The chiral branches that
# flow across xi=0 mirror the edge states of the model: there are 2|C| of
# them, |C| per entanglement boundary, and the ring has two boundaries.

import numpy as np
from pyqula import geometry

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.add_haldane(0.1) # Haldane flux, opens a topological gap

print("Chern number =",round(h.get_chern(nk=20),4))

(ks,xis) = h.get_entanglement_spectrum(nsuper=10, # cells in the ring
                                       nk=101 # parallel momenta
                                       )

# entropy per parallel unit cell (it counts both entanglement boundaries)
print("Entanglement entropy per unit cell =",
      round(h.get_entanglement_entropy(nsuper=10,nk=20),4))

# the same model with a large sublattice mass is trivial and gapped
h2 = g.get_hamiltonian(has_spin=False)
h2.add_sublattice_imbalance(0.6)
(ks2,xis2) = h2.get_entanglement_spectrum(nsuper=10,nk=101)

import matplotlib.pyplot as plt

for (i,(k,x,name)) in enumerate([(ks,xis,"Haldane model, C = 1"),
                                 (ks2,xis2,"Trivial insulator, C = 0")]):
    plt.subplot(1,2,i+1)
    for j in range(x.shape[1]): plt.plot(k,x[:,j],c="blue",marker="o",
                                         markersize=2,linestyle="none")
    plt.axhline(0.,c="black",linewidth=0.5)
    plt.ylim([-8,8])
    plt.xlabel("$k_\\parallel$") ; plt.ylabel("Entanglement spectrum $\\xi_n$")
    plt.title(name)

plt.tight_layout()
plt.show()
