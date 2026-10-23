# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Three-band model of a transition metal dichalcogenide monolayer, here
# WSe2, from Liu et al., PRB 88, 085433 (2013). The basis is the d_z2,
# d_xy and d_x2-y2 orbitals of the metal atom on a triangular lattice,
# with hoppings up to the third neighbors and the spin-orbit coupling
# (lambda/2) Lz sz of the metal atom. Energies are in eV and lengths in
# Angstrom. The valence band at K is split by 2 lambda, with opposite spin
# at K and K': the spin-valley locking.

import numpy as np
from pyqula import specialhamiltonian

h = specialhamiltonian.three_band_tmdc("WSe2") # TNN model, GGA, with SOC
print("gap in eV ",h.get_gap())
(k,e,sz) = h.get_bands(operator="sz") # bands, and the spin of each state

import matplotlib.pyplot as plt

plt.scatter(k,e,c=sz,s=4,cmap="bwr",vmin=-1,vmax=1)
plt.xticks([0,k.max()/3,k.max()/2,2*k.max()/3,k.max()],
           ["$\\Gamma$","K","M","K'","$\\Gamma$"])
plt.ylabel("Energy [eV]")
plt.colorbar(label="$\\langle S_z \\rangle$")
plt.tight_layout()
plt.show()
