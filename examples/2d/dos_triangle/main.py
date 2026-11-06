# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Density of states of graphene without broadening. A plain sum over the
# k-mesh needs one Lorentzian per k-point, so with a small delta it shows
# the single k-points as spikes. The triangle integration takes the bands as
# linear inside each triangle of the mesh and integrates exactly, so the
# same mesh gives the DOS at delta = 0: the linear DOS at the Dirac point
# and the logarithmic van Hove peaks at E = +-t. Here nk is odd and a
# multiple of 3, so that no triangle is flat at the van Hove energy.
# The sz-projected DOS of a spinful model uses the same mesh.

import numpy as np
from pyqula import geometry

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)

es = np.linspace(-3.5,3.5,701) # energies
(es,dg) = h.get_dos(energies=es,nk=201,delta=1e-3,write=False) # spikes
(es,dt) = h.get_dos(energies=es,nk=201,delta=0.,integration="triangle",
                    write=False) # exact limit

# spin-split graphene: the difference of the two spin channels
hs = g.get_hamiltonian(has_spin=True)
hs.add_exchange([0.,0.,0.4])
(es,dsz) = hs.get_dos(energies=es,nk=201,delta=0.,integration="triangle",
                      operator="sz",write=False)

import matplotlib.pyplot as plt
plt.plot(es,dg,c="gray",lw=0.8,label="grid, delta=1e-3")
plt.plot(es,dt,c="blue",label="triangle, delta=0")
plt.plot(es,dsz,c="red",label="triangle, sz, J=0.4")
plt.xlabel("Energy [t]") ; plt.ylabel("DOS [states/t/cell]")
plt.legend()
plt.show()
