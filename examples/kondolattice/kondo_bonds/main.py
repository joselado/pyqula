# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Kondo lattice with a coupling between different sites. Each moment is
# coupled to the conduction electrons of the two neighboring sites, which
# are in the neighboring unit cells of the chain, and each of these Kondo
# bonds gets its own hybridization field V_b = -(J_b/2) <c^dag f>. With one
# moment per cell the bonds give a hybridization that depends on k,
# V(k) = sum_R V_R exp(ikR).

import numpy as np
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

hc = geometry.chain().get_hamiltonian(has_spin=True) # conduction electrons
h = KondoLatticeHamiltonian(hc)

def J(rc, rm): # coupling of the conduction site at rc to the moment at rm
    d = np.linalg.norm(rc - rm)
    if abs(d - 1.0) < 1e-6: return 2.0 # first neighbors
    return 0.0 # no onsite coupling here

h2 = h.get_mean_field_hamiltonian(J=J, filling=0.5, nk=200, mf=(1.0, 0.0))

print("Kondo bonds (conduction site, moment, cell of the site, coupling)")
for b, V in zip(h2.kondo_bonds, h2.hybridization):
    print(b, "V =", np.round(V, 4))
print("<n_f> per moment (target 1.0):", h2.local_occupation)

# V(k) of the two bonds: odd in k, so it vanishes at k=0 and k=pi, where
# the f-level stays at the chemical potential
for k in [0., 0.25, 0.5]: # in units of 2 pi
    Vk = sum(V*np.exp(2j*np.pi*k*b[2][0])
             for b, V in zip(h2.kondo_bonds, h2.hybridization))
    print("k =", k, "V(k) =", np.round(Vk, 4))

h2.get_bands(operator="zposition") # weight on the f-sites, in BANDS.OUT
