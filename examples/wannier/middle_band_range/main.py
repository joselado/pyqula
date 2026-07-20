# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
from pyqula import geometry

# get_wannier_hamiltonian's bands=[a,b] argument selects a contiguous
# range of bands (0-indexed into eigh's ascending output at every
# k-point), not necessarily the lowest ones -- e.g. bands=[3,4] out of a
# 6-band model Wannierizes bands 3 and 4 together, skipping bands 0-2
# and 5. All bands in the range are always Wannierized jointly, as one
# group -- get_wannier_hamiltonian never silently splits the selection
# into independent pieces (an optional auto_split_clusters=True exists
# for band groups you know decompose into topologically trivial pieces,
# but is not the default: a jointly-trivial selection can still contain
# individually topologically-obstructed sub-pieces, e.g. two gapped
# sub-bands with opposite nonzero Chern number that only cancel when
# treated jointly, so splitting is never done automatically).

# a genuine 3-site unit cell (distinct, non-periodically-repeating onsite
# energies) with Rashba SOC and an exchange field: 3 sites/cell x 2 spin
# = 6 bands per k-point
g = geometry.chain().get_supercell(3)
h = g.get_hamiltonian(has_spin=True)
h.add_onsite([0.3,2.1,4.7]) # distinct onsite energy per site
h.add_rashba(0.3) # Rashba spin-orbit coupling
h.add_exchange([0.1,0.15,0.2]) # exchange field

# Wannierize bands 3,4 jointly -- a middle pair of the spectrum, not the
# lowest 2
hwan = h.get_wannier_hamiltonian(bands=[3,4],nk=24,num_iter=3000)

print("Selected bands:",hwan.wannier_band_indices)
print("Number of Wannier functions:",hwan.intra.shape[0])
print("Wannier centres (Cartesian):\n",hwan.wannier_centres)
print("Total spread Omega:",hwan.wannier_spread_total)

# compare the original bands 3-4 against the Wannierized model's bands,
# along the same k-path
(k,e) = h.get_bands(write=False)
(kw,ew) = hwan.get_bands(write=False)

import matplotlib.pyplot as plt
plt.scatter(k,e,c="black",s=20,label="original bands")
plt.scatter(kw,ew,c="red",s=6,label="Wannierized bands 3-4")
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.legend()
plt.show()
