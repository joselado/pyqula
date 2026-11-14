# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Chern number of the magnon bands of a ferromagnet, from time-dependent
# Hartree-Fock.
#
# A magnon at momentum Q is a sum of electron-hole pairs with a spin flip,
# an electron at k+Q and a hole at k. Written in the orbitals,
#   Phi_Q(k;a,b) = sum_vc X_vc(k) u_c,k+Q(a) conj(u_v,k(b)) ,
# it does not depend on the phases of the Bloch states, and it plays the
# role of the Bloch state of a band. The Chern number of a magnon band is
# the sum of its Berry fluxes through the plaquettes of a mesh of Q, over
# 2 pi, with the sign of h.get_chern().
#
# The magnet is the honeycomb lattice with a second-neighbor hopping 0.3,
# a Kane-Mele coupling 0.15 and a Zeeman field 3, with U=8 at half
# filling. The field polarizes it along -z and gaps the Goldstone mode, and
# the Kane-Mele coupling times the second-neighbor hopping acts as a
# Dzyaloshinskii-Moriya term for the magnons. The two magnon bands have
# C=+1 and C=-1, and the two together C=0; the opposite Kane-Mele coupling
# flips both.

import numpy as np
from pyqula import geometry

nk = 8 # the mean field and the magnon problem share this mesh
g = geometry.honeycomb_lattice()
for km in [0.15,-0.15]:
    h = g.get_hamiltonian(tij=[1.0,0.3]) # first and second neighbor hopping
    h.add_kane_mele(km) # spin-orbit coupling, conserves Sz
    h.add_zeeman([0.,0.,3.0]) # the field gaps the magnons
    hmf = h.get_mean_field_hamiltonian(U=8.0,filling=0.5,mf="ferro",nk=nk)
    print("Kane-Mele",km,"| Sz per site",np.round(hmf.get_vev("sz").real,6))
    for bands in [[0],[1],[0,1]]:
        c = hmf.get_magnon_chern(bands=bands,nq=8,nk=nk)
        print("   bands",bands,"C =",np.round(c,8))

# The record says if the integer can be trusted: the band has to be
# separated from the other one (gap) and below the Stoner continuum
# (bound), and the largest Berry flux of a plaquette has to stay below pi.
# For the lower band of the first magnet it gives a gap of 0.33, a bound of
# 2.75 and a largest flux of 0.54, about 0.17 pi.
h = g.get_hamiltonian(tij=[1.0,0.3])
h.add_kane_mele(0.15)
h.add_zeeman([0.,0.,3.0])
hmf = h.get_mean_field_hamiltonian(U=8.0,filling=0.5,mf="ferro",nk=nk)
r = hmf.get_magnon_chern(bands=[0],nq=8,nk=nk,full_output=True)
print("lower band from",np.round(r.energies.min(),4),"to",
      np.round(r.energies.max(),4))
print("gap",np.round(r.gap,4),"| bound",np.round(r.bound,4),
      "| largest flux / pi",np.round(r.max_flux/np.pi,4))
# the same integer on a finer mesh of Q
print("nq=12: C =",np.round(hmf.get_magnon_chern(bands=[0],nq=12,nk=nk),8))
