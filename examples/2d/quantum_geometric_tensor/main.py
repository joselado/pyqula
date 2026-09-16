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

# The full non-Abelian tensor, returned in the orbital basis (sum over the
# occupied pair of |u_m> Q^{mn} <u_n|), which does not depend on the basis
# the diagonalization picks inside the degenerate pair. With no spin-orbit
# coupling it is block diagonal in spin, and the trace over the spin-up
# orbitals (0 and 2) is the contribution of the spin-up electrons
inds,g_na,omega_na = topology.quantum_geometric_tensor_path(h,occ_idxs=[0,1],
        nk=200,non_abelian=True)
up,dn = [0,2],[1,3] # spin-orbital order: site 0 up/down, site 1 up/down
spin_trace = lambda T,o: T[...,o,o].sum(axis=-1) # trace over orbitals o

# The Abelian (band-trace) quantum metric/Berry curvature is exactly the
# trace of the non-Abelian tensor -- no need for a second, separate k-path
# sweep to get it
g_ab = spin_trace(g_na,up) + spin_trace(g_na,dn)
omega_ab = spin_trace(omega_na,up) + spin_trace(omega_na,dn)

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(inds,omega_ab[:,0,1].real,label="Abelian (trace)")
plt.plot(inds,spin_trace(omega_na,up)[:,0,1].real,label="spin up")
plt.plot(inds,spin_trace(omega_na,dn)[:,0,1].real,label="spin down",linestyle="dashed")
plt.xlabel("kpath") ; plt.xticks([]) ; plt.ylabel("Berry curvature")
plt.legend()

plt.subplot(1,2,2)
plt.plot(inds,g_ab[:,0,0].real,label="Abelian (trace)")
plt.plot(inds,spin_trace(g_na,up)[:,0,0].real,label="spin up")
plt.plot(inds,spin_trace(g_na,dn)[:,0,0].real,label="spin down",linestyle="dashed")
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
