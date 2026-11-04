# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Bott index of the Haldane model and spin Bott index of the Kane-Mele model
# with Rashba coupling, on periodic supercells with Anderson disorder. The
# clean value is the Chern number (spin Chern number) of the k-space
# routines; weak disorder keeps it, strong disorder localizes every state
# and gives zero, the Anderson insulator
import numpy as np
from pyqula import geometry

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False) # spinless Hamiltonian
h.add_haldane(0.15) # Haldane coupling, C=1
print("Chern number in k-space =",round(h.get_chern(nk=30),3))
hs = h.get_supercell(10) # periodic supercell of 200 sites
for W in [0.,2.,4.,6.,8.]: # strength of the disorder
    bs = [] # Bott index of each disordered sample
    for seed in range(5):
        rng = np.random.default_rng(seed) # random numbers
        hd = hs.copy()
        hd.add_onsite(W*(rng.random(len(hd.geometry.r))-0.5)) # Anderson disorder
        bs.append(round(hd.get_bott_index(),3)) # Bott index
    print("W =",W,"  B =",bs)

h = g.get_hamiltonian(has_spin=True) # spinful Hamiltonian
h.add_kane_mele(0.1) # spin-orbit coupling
h.add_rashba(0.05) # Rashba coupling, s_z is not conserved
for m in [0.,0.4,0.8]: # sublattice imbalance
    hm = h.copy()
    hm.add_sublattice_imbalance(m) # trivial for a large imbalance
    hs = hm.get_supercell(6) # periodic supercell of 72 sites
    print("imbalance =",m,"  Bs =",round(hs.get_spin_bott_index(),3),
          "  B =",round(hs.get_bott_index(),3))
