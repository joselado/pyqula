# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
from pyqula import topology
import numpy as np

# Multiorbital/multiband quantum geometric tensor of the Haldane model,
# via the exact sum-over-states Kubo formula (see topologytk/qgt.py)

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian() # spinful by default: 4 bands, two exactly
                         # spin-degenerate pairs (no spin-orbit coupling)
h.add_haldane(0.2)
h.shift_fermi(0.3) # put the Fermi level safely mid-gap (gap is [-0.9,0.9])

# The full non-Abelian tensor resolves the two occupied (spin up/down)
# bands individually: with no spin-orbit coupling the two spin channels do
# not mix, so the tensor is block diagonal, and each diagonal block equals
# the single-band result of the corresponding spinless problem
inds,g_na,omega_na = topology.quantum_geometric_tensor_path(h,occ_idxs=[0,1],
        nk=200,non_abelian=True)

# The Abelian (band-trace) quantum metric/Berry curvature is exactly the
# trace of the non-Abelian tensor over the occupied-band indices -- no
# need for a second, separate k-path sweep to get it
g_ab = g_na[:,:,:,0,0] + g_na[:,:,:,1,1]
omega_ab = omega_na[:,:,:,0,0] + omega_na[:,:,:,1,1]

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(inds,omega_ab[:,0,1].real,label="Abelian (trace)")
plt.plot(inds,omega_na[:,0,1,0,0].real,label="band 0")
plt.plot(inds,omega_na[:,0,1,1,1].real,label="band 1",linestyle="dashed")
plt.xlabel("kpath") ; plt.xticks([]) ; plt.ylabel("Berry curvature")
plt.legend()

plt.subplot(1,2,2)
plt.plot(inds,g_ab[:,0,0].real,label="Abelian (trace)")
plt.plot(inds,g_na[:,0,0,0,0].real,label="band 0")
plt.plot(inds,g_na[:,0,0,1,1].real,label="band 1",linestyle="dashed")
plt.xlabel("kpath") ; plt.xticks([]) ; plt.ylabel("Quantum metric g_xx")
plt.legend()

plt.tight_layout()
plt.show()

# Cross-check: integrating the Berry curvature over the BZ gives the
# same (quantized) Chern number as the independent Wilson-loop method
c_wilson = topology.chern(h,nk=20)
c_qgt = topology.chern_from_qgt(h,nk=20,occ_idxs=[0,1])
print("Chern number (Wilson loop):",c_wilson)
print("Chern number (quantum geometric tensor):",c_qgt)
