# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# The two calculations of arXiv:2302.09701 for the Kondo lattice mean field,
# on the square lattice instead of the chain of the paper:
#  1) the spectral functions of the conduction electron c and of the
#     composite fermion F = (2/3)(sigma.S) c along a k-path;
#  2) the spin susceptibility of the local moments at a momentum q, in the
#     mean field and with a residual interaction U' in the RPA.
# Nothing changes in the calls with the dimensionality: the k-path and the
# momentum q are given in units of the reciprocal lattice vectors.
#
# The paper writes the Kondo term as J' (c^dag sigma c).S, and the loop of
# pyqula takes J = 2 J'.

import numpy as np
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

g = geometry.square_lattice()
hc = g.get_hamiltonian(has_spin=True) # eps_k = 2 (cos kx + cos ky)
h = KondoLatticeHamiltonian(hc)
Jp = 2.0 # J' in units of the hopping
# half filling of a band with particle-hole symmetry: the Kondo insulator
h2 = h.get_mean_field_hamiltonian(J=2*Jp, filling=0.5, nk=40,
        mf=([Jp+0j], [0.0]), maxerror=1e-7)
V = h2.hybridization[0].real
print("hybridization V, f occupation, Lagrange multiplier")
print(V, h2.local_occupation[0], h2.constraint_lambda[0])
print("direct gap 2V and indirect gap 2 (sqrt(4 + V^2) - 2)")
print(2*V, 2*(np.sqrt(4+V**2)-2))

# 1) spectral functions along Gamma, X, M, Gamma
nseg = 50 # k-points of each segment of the path
corners = np.array([[0., 0., 0.], [0.5, 0., 0.], [0.5, 0.5, 0.], [0., 0., 0.]])
kpath = np.concatenate([np.linspace(corners[i], corners[i+1], nseg,
        endpoint=False) for i in range(3)])
energies = np.linspace(-6., 6., 301)
(ks, es, A) = h2.get_composite_spectral_function(kpath=kpath,
        energies=energies, delta=0.05)
up = es > 0
for (label, ik) in [("Gamma", 0), ("X", nseg), ("M", 2*nseg)]:
    print(label, ": upper band at", es[up][np.argmax(A["FF"][ik][up])],
            ", weight of F over weight of c there:",
            np.max(A["FF"][ik][up])/np.max(A["cc"][ik][up]))
kidx = np.repeat(np.arange(len(ks)), len(es))
np.savetxt("COMPOSITE_SPECTRAL_FUNCTION.OUT", np.array([kidx,
        np.tile(es, len(ks)), A["cc"].ravel(), A["FF"].ravel()]).T)

# 2) spin susceptibility of the moments at Gamma, X and M
ws = np.linspace(0., 6., 301)
for (label, q) in [("Gamma", [0., 0., 0.]), ("X", [0.5, 0., 0.]),
        ("M", [0.5, 0.5, 0.])]:
    (e, chi0) = h2.get_moment_spinchi(q=q, energies=ws, nk=40, delta=0.05)
    (e, chi) = h2.get_moment_spinchi(q=q, energies=ws, nk=40, delta=0.05,
            U=-2.0)
    (e, chis) = h2.get_moment_spinchi(q=q, energies=[0.0], nk=40, delta=0.05)
    print(label, ": maximum of Im chi at w =", ws[np.argmax(chi0[:, 0, 0].imag)],
            "(mean field) and at w =", ws[np.argmax(chi[:, 0, 0].imag)],
            "(U' = -2 t); 1 - U' chi(q,0) =", 1+2.0*chis[0, 0, 0].real)
