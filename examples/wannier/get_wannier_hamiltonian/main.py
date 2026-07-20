# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
from pyqula import geometry

# get_wannier_hamiltonian computes maximally-localized Wannier functions for
# a subset of a Hamiltonian's bands (via the wannierpy pure-Python Wannier90
# port bundled at src/pyqula/wanniertk/wannierpy/) and returns a new,
# smaller Hamiltonian defined purely by the resulting real-space hoppings.

# staggered honeycomb lattice: a sublattice potential opens a gap, giving a
# genuinely dispersive (non-trivial) valence band to Wannierize
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.add_onsite([0.8,-0.8])

# Wannierize just the lowest (valence) band -- bands=[0,0] selects the
# lowest band (band index 0) at every k-point of the wannierization mesh
# (nk points per periodic direction); no disentanglement, since the band
# is pre-selected
hwan = h.get_wannier_hamiltonian(bands=[0,0],nk=12)

print("Number of Wannier functions:",hwan.intra.shape[0])
print("Wannier centres (Cartesian):\n",hwan.wannier_centres)
print("Wannier spreads:",hwan.wannier_spreads)
print("Total spread Omega:",hwan.wannier_spread_total)

# compare the original valence band against the Wannierized model's band,
# along the same k-path
(k,e) = h.get_bands(write=False)
(kw,ew) = hwan.get_bands(write=False)

import matplotlib.pyplot as plt
plt.scatter(k,e,c="black",s=20,label="original bands")
plt.scatter(kw,ew,c="red",s=6,label="Wannierized valence band")
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.legend()
plt.show()
