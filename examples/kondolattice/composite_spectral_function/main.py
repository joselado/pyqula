# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Spectral function of the Kondo lattice resolved in momentum, for the
# conduction electron c and for the composite fermion
#   F = (2/3) (sigma.S) c ,
# an electron together with a spin flip of the moment of the same site.
# Chen, Stoudenmire, Komijani and Coleman, arXiv:2302.09701, compute the
# 2x2 matrix of Green functions of (c,F) with matrix product states and
# compare it with the mean field of the auxiliary fermions. Here we compute
# the mean-field side.
#
# That work writes the Kondo term as J' (c^dag sigma c).S, and the
# self-consistent loop of pyqula takes the Coqblin-Schrieffer coupling
# J = 2 J'. In the mean field the Kondo term is V f^dag c + h.c., and the
# equation of motion of c gives F = (4 V^*/(3J)) f.
#
# We check three things:
#  1) the loop at half filling gives the V of the T=0 gap equation
#       1/J' = 1/N sum_k 1/sqrt((eps_k/2)^2 + V^2) ;
#  2) the Green functions obey the relations of the exact problem,
#       G_cc = g_c + g_c (3J'/2) G_FF (3J'/2) g_c ,  G_cF = g_c (3J'/2) G_FF ,
#     and the inverse of the 2x2 matrix has 1/g_F in its FF element;
#  3) at strong coupling V = J', and the exact energy to add an electron
#     to one site is 3J'/2. This is why the paper multiplies V by 3/2.

import numpy as np
from scipy.optimize import brentq
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

gc = geometry.chain()
hc = gc.get_hamiltonian(has_spin=True) # conduction electrons, eps_k = 2 cos k
h = KondoLatticeHamiltonian(hc)

Jp = 2.0 # J' of the paper, in units of the hopping
# half filling is the Kondo insulator: the chemical potential and the
# Lagrange multiplier are zero by particle-hole symmetry
h2 = h.get_mean_field_hamiltonian(J=2*Jp, filling=0.5, nk=400,
        mf=([Jp+0j], [0.0]), maxerror=1e-8)
V = h2.hybridization[0].real

# 1) the gap equation
k = 2*np.pi*(np.arange(4000)+0.5)/4000
eps = 2*np.cos(k)
V0 = brentq(lambda v: np.mean(1/np.sqrt((eps/2)**2+v**2))-1/Jp, 1e-6, 50)
print("hybridization V of the loop, and of the gap equation")
print(V, V0)
print("f occupation and Lagrange multiplier")
print(h2.local_occupation[0], h2.constraint_lambda[0])

# 2) the matrix Green function and its relations
delta = 0.05
energies = np.linspace(-6., 6., 301)
(ks, es, G) = h2.get_composite_spectral_function(energies=energies, nk=101,
        delta=delta, J=2*Jp, green=True)
z = es+1j*delta
err = 0.0
for ik in range(len(ks)):
    gck = 1/(z-2*np.cos(2*np.pi*ks[ik][0])) # bare conduction propagator
    v = 1.5*Jp # hybridization of c with the composite fermion
    err = max(err, np.max(np.abs(G["cc"][ik]-(gck+gck*v*G["FF"][ik]*v*gck))))
    err = max(err, np.max(np.abs(G["cF"][ik]-gck*v*G["FF"][ik])))
    # the FF element of the inverse matrix is 1/g_F
    det = G["cc"][ik]*G["FF"][ik]-G["cF"][ik]*G["Fc"][ik]
    err = max(err, np.max(np.abs(det/G["cc"][ik]-G["gF"][ik])))
print("largest violation of the three relations")
print(err)
print("weight of the composite fermion |4V/(3J)|^2, and integral of A_FF")
(ks, es, A) = h2.get_composite_spectral_function(energies=energies, nk=101,
        delta=delta, J=2*Jp)
print((2*V/(3*Jp))**2, np.trapezoid(A["FF"][0], es))

# 3) strong coupling: the bands at eps_k = 0 are at +-V
hs = h.get_mean_field_hamiltonian(J=2*10.0, filling=0.5, nk=200,
        mf=([10.0+0j], [0.0]), maxerror=1e-8)
print("V/J' at J' = 10 t (1 at strong coupling)")
print(hs.hybridization[0].real/10.0)
es = np.linspace(0., 20., 2001)
for scale in [1.0, 1.5]:
    (ks, e, A) = hs.get_composite_spectral_function(kpath=[[0.25, 0., 0.]],
            energies=es, delta=0.02, hybridization_scale=scale)
    print("energy of the upper band over J', hybridization_scale =", scale)
    print(es[np.argmax(A["FF"][0])]/10.0)
print("exact energy to add an electron to one site, over J'")
print(1.5)

# the maps of the figure of the paper: A_cc and A_FF over (k, energy),
# with the hybridization multiplied by 3/2
energies = np.linspace(-6., 6., 301)
(ks, es, A) = h2.get_composite_spectral_function(energies=energies, nk=201,
        delta=delta, hybridization_scale=1.5)
kk, ee = np.meshgrid(ks[:, 0], es, indexing="ij")
np.savetxt("COMPOSITE_SPECTRAL_FUNCTION.OUT", np.array([kk.ravel(),
        ee.ravel(), A["cc"].ravel(), A["FF"].ravel(),
        A["gF"].real.ravel(), A["gF"].imag.ravel()]).T)
