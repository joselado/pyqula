# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
from pyqula import geometry

# get_wannier_hamiltonian computes maximally-localized Wannier functions for
# a subset of a Hamiltonian's bands (via the wannierpy package -- see
# vendor/wannierpy in this repo, "pip install -e vendor/wannierpy" to enable
# this feature) and returns a new, smaller Hamiltonian defined purely by the
# resulting real-space hoppings. This example wannierizes a spinful model,
# where each site contributes two spin-orbitals to the Hamiltonian basis.

# dimerized chain: two sites per unit cell (sublattices A/B), so a
# staggered onsite potential gives a genuine sublattice imbalance
g = geometry.bichain()
h = g.get_hamiltonian(has_spin=True) # spinful, needed for Rashba/exchange
h.add_onsite([0.8,-0.8]) # sublattice imbalance (one value per site)
h.add_rashba(0.4) # Rashba spin-orbit coupling
h.add_exchange([0.,0.,0.3]) # exchange (Zeeman-like) field

# 2 sites x 2 spin = 4 bands per k-point; Wannierize the lowest 2
# (num_iter raised above the default 200 -- the spin-orbit-coupled spread
# minimization needs more CG steps to converge than a spinless case)
hwan = h.get_wannier_hamiltonian(num_bands=2,nk=24,num_iter=1000)

print("Number of Wannier functions:",hwan.intra.shape[0])
print("Wannier centres (Cartesian):\n",hwan.wannier_centres)
print("Wannier spreads:",hwan.wannier_spreads)
print("Total spread Omega:",hwan.wannier_spread_total)

# compare the original lowest two bands against the Wannierized model's
# bands, along the same k-path
(k,e) = h.get_bands(write=False)
(kw,ew) = hwan.get_bands(write=False)

import matplotlib.pyplot as plt
plt.scatter(k,e,c="black",s=20,label="original bands")
plt.scatter(kw,ew,c="red",s=6,label="Wannierized lowest 2 bands")
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.legend()
plt.show()
