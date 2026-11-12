# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# The two calculations of arXiv:2302.09701 for the Kondo lattice mean field,
# on a finite system: a honeycomb flake of 24 sites, each one with a local
# moment. A finite system has no momentum, so
#  1) the spectral functions of the conduction electron c and of the
#     composite fermion F = (2/3)(sigma.S) c are the local ones of each
#     site, what a tunneling experiment sees;
#  2) the spin susceptibility is a matrix between the 24 moments, and the
#     momentum q is ignored.
# The paper writes the Kondo term as J' (c^dag sigma c).S, and the loop of
# pyqula takes J = 2 J'.

import numpy as np
from pyqula import islands
from pyqula.kondolattice import KondoLatticeHamiltonian

g = islands.get_geometry(name="honeycomb", n=2, nedges=6) # 24 sites, 0D
hc = g.get_hamiltonian(has_spin=True)
h = KondoLatticeHamiltonian(hc)
n = len(g.r)
Jp = 2.0 # J' in units of the hopping
h2 = h.get_mean_field_hamiltonian(J=2*Jp, filling=0.5,
        mf=([Jp+0j]*n, [0.0]*n), maxerror=1e-7)
V = h2.hybridization.real
print("number of Kondo sites:", n)
print("hybridization V, smallest and largest (the edge is different)")
print(np.min(V), np.max(V))
print("largest deviation of the f occupation from one")
print(np.max(np.abs(h2.local_occupation-1.0)))

# 1) local spectral functions of every site
energies = np.linspace(-6., 6., 601)
(ks, es, A) = h2.get_composite_spectral_function(energies=energies,
        delta=0.05, J=2*Jp, site=list(range(n)))
print("shape of A_FF: (sites, one k-point, energies)")
print(A["FF"].shape)
levels = np.linalg.eigvalsh(np.array(h2.intra)) # levels of the flake
print("gap of the flake, between the last filled and the first empty level")
print(np.min(levels[levels > 0])-np.max(levels[levels < 0]))
print("weight of F on each site, |4V/(3J)|^2, smallest and largest")
print(np.min((2*V/(3*Jp))**2), np.max((2*V/(3*Jp))**2))

# 2) spin susceptibility between the moments
ws = np.linspace(0., 6., 301)
(e, chi0) = h2.get_moment_spinchi(energies=ws, delta=0.05)
(e, chi) = h2.get_moment_spinchi(energies=ws, delta=0.05, U=-2.0)
print("shape of chi: (energies, moments, moments)")
print(chi.shape)
# the staggered response, the one of an antiferromagnetic field
sign = np.array(g.sublattice) # +1 and -1 on the two sublattices
stag0 = np.einsum("i,wij,j->w", sign, chi0, sign)/n
stag = np.einsum("i,wij,j->w", sign, chi, sign)/n
print("maximum of the staggered Im chi, mean field and U' = -2 t, at w =")
print(ws[np.argmax(stag0.imag)], ws[np.argmax(stag.imag)])
