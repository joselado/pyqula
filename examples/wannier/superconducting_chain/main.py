# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
from pyqula import geometry

# get_wannier_hamiltonian also supports superconducting (Nambu/BdG,
# h.has_eh=True) Hamiltonians, and *enforces* electron-hole (particle-hole)
# symmetry on the result: the returned Hamiltonian's real-space hoppings
# satisfy C @ conj(h_R) @ C^-1 == -h_R exactly, for the unitary operator
# C stored as hwan.wannier_particle_hole_operator. Because a Nambu
# spectrum is symmetric under E -> -E, bands=[a,b] must be closed under
# the pairing n -> num_orbitals-1-n (picking a band forces picking its
# exact particle-hole partner), i.e. centred on the gap
# (a+b == num_orbitals-1).

g = geometry.chain()
h = g.get_hamiltonian(has_spin=True) # spinful, needed for Rashba/exchange
h.add_rashba(0.5) # Rashba spin-orbit coupling
h.add_exchange([0.,0.,0.3]) # exchange field
h.shift_fermi(1.0) # move away from particle-hole symmetric filling
h.add_swave(0.2) # s-wave superconducting pairing -- turns h.has_eh on

# 1 site x 2 spin x 2 (electron/hole) = 4 bands per k-point; Wannierize
# bands 1,2, the 2 bands nearest the gap (the low-energy Bogoliubov
# quasiparticle branch) -- the only electron-hole-symmetric pair
# available besides the full 4-band manifold
hwan = h.get_wannier_hamiltonian(bands=[1,2],nk=24,num_iter=1000)

print("Wannierized bands:",hwan.wannier_band_indices)
print("Number of Wannier functions:",hwan.intra.shape[0])
print("Total spread Omega:",hwan.wannier_spread_total)

# verify the enforced electron-hole symmetry directly: C * conj(h_R) *
# C^-1 == -h_R for every real-space hopping matrix of the result
C = hwan.wannier_particle_hole_operator
Cinv = np.linalg.inv(C)
violation = np.max(np.abs(C@np.conjugate(hwan.intra)@Cinv + hwan.intra))
for hop in hwan.hopping:
    violation = max(violation,np.max(np.abs(C@np.conjugate(hop.m)@Cinv + hop.m)))
print("Max electron-hole symmetry violation:",violation)

# compare the original bands 1,2 against the Wannierized model's bands
(k,e) = h.get_bands(write=False)
(kw,ew) = hwan.get_bands(write=False)

import matplotlib.pyplot as plt
plt.scatter(k,e,c="black",s=20,label="original BdG bands")
plt.scatter(kw,ew,c="red",s=6,label="Wannierized bands (PHS-enforced)")
plt.axhline(0,c="grey",lw=0.5)
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.legend()
plt.show()
