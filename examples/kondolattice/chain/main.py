# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Abrikosov-pseudofermion (Read-Newns) mean-field theory for the Kondo
# lattice / periodic Anderson model -- the standard minimal model of heavy
# fermion compounds. Follows P. Coleman, "Heavy Fermions: electrons at the
# edge of magnetism", arXiv:cond-mat/0612006, Sec. III.C (Eq. 65-99): each
# localized moment is represented as S_j = 1/2 f_j^dagger sigma f_j
# (Abrikosov pseudofermions, one per site of the conduction-electron
# geometry, offset in z), exchange-coupled to the conduction electron at
# the same site, and self-consistently decoupled into a hybridization
# field V_j = -(J/2) <f_j^dagger c_j> plus a Lagrange multiplier lam_j
# enforcing exactly one f-fermion per site.
#
# V=0 is always itself a self-consistent solution (like the trivial root
# of the BCS gap equation) -- an unseeded run stays there, so a nonzero
# mf=(V, lam) seed is used below to find the other, genuinely hybridized
# solution; see pyqula.kondolattice's module docstring for why this branch
# is the true (lower-energy) ground state where both coexist, and for why
# the onset in J is first-order-like at any finite smearing T rather than
# the continuous exponential T=0 large-N theory predicts.

import numpy as np
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

gc = geometry.chain()
hc = gc.get_hamiltonian(has_spin=True) # conduction electrons, t=1 chain
h = KondoLatticeHamiltonian(hc)

# filling=0.15 keeps the lattice-wide chemical potential inside the
# dispersing conduction band, away from the bare f-sector's macroscopically
# degenerate flat band at V=0 (see the module docstring's filling caveat)
seed = ([0.3+0.0j], [0.0]) # (V, lam) -- see the module docstring above
h2, etot = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=200,
        mf=seed, mix=0.3, maxerror=1e-6, maxite=3000,
        return_total_energy=True)
if h2 is None:
    raise RuntimeError("SCF did not converge")

print("Converged <n_f> per localized site (target: 1.0)")
print(h2.local_occupation)
print("Converged hybridization V per localized site")
print(h2.hybridization)
print("Converged per-site Lagrange multiplier")
print(h2.constraint_lambda)
print("Ground-state energy per unit cell")
print(etot)
print("Direct hybridization gap 2|V| (Coleman Eq. 88)")
print(2*abs(h2.hybridization[0]))

# the f-sublattice sits at z=+1.0 (see KondoLatticeHamiltonian.__init__),
# the conduction sublattice at the original z -- project onto it to see
# which band is "heavy" (mostly f character) in the printed bands.OUT
h2.get_bands(operator=lambda r: 1.*(r[2] > 0.5))
