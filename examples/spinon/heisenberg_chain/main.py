# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Abrikosov-pseudofermion (RVB) mean-field theory for the antiferromagnetic
# Heisenberg chain. Represents each spin as S_i = 1/2 f_i^dagger sigma f_i
# (Savary & Balents, "Quantum Spin Liquids: a review", arXiv:1601.03742,
# Sec. 4) and self-consistently decouples J S_i.S_j into an RVB bond order
# parameter chi_ij = <f_i^dagger f_j>, enforcing exactly one auxiliary
# fermion per site at every site (not just on lattice average).
#
# The chain (a 1-site unit cell, translationally uniform) has a unique
# self-consistent RVB solution, so this example prints a single well-
# defined answer. A frustrated lattice (e.g. triangular, kagome) can
# instead have several distinct self-consistent parton ansatze (different
# flux sectors) at the same J -- which one a random mf= guess converges to
# is then itself part of the physics, not a bug; see the user guide
# ("Abrikosov-pseudofermion (spinon) mean field for Heisenberg models").

import numpy as np
from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian

np.random.seed(0) # reproducible: an unseeded random mf guess can
                   # occasionally fail to converge within maxite

g = geometry.chain()
h = SpinonHamiltonian(g) # zero bare hopping -- a pure spin model

# return_total_energy=True is needed for the correct mean-field ground-
# state energy: h2.get_total_energy() alone is only the sum of occupied
# spinon-band energies, missing the Hartree-Fock double-counting
# correction (and the constraint's own grand-potential term).
h2, etot = h.get_mean_field_hamiltonian(J1=1.0, nk=24, mix=0.3,
        maxerror=1e-5, maxite=2000, return_total_energy=True)
if h2 is None:
    raise RuntimeError("SCF did not converge")

print("Converged <n_i> per site (target: 1.0)")
print(h2.local_occupation)
print("Converged per-site Lagrange multiplier (local chemical potential)")
print(h2.constraint_lambda)
print("Ground-state energy per site")
print(etot/len(g.r))

h2.get_bands(operator="sz") # write the spinon dispersion to bands.OUT
