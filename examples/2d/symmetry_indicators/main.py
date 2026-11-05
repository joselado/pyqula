# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Topological invariants from the symmetry eigenvalues of the occupied states
# at a few momenta. First the Chern number modulo 3 of the Haldane model, from
# the C3 eigenvalues at Gamma, K and K', as a sublattice imbalance closes the
# gap. Then the Z2 invariant of the Kane-Mele model from the inversion
# eigenvalues at the four time-reversal-invariant momenta, as the hopping of
# one of the three bonds grows and dimerizes the lattice.
import numpy as np
from pyqula import geometry

g = geometry.honeycomb_lattice() # honeycomb lattice
for M in [0.,0.3,0.6,0.9,1.2]: # sublattice imbalance, in units of t
    h = g.get_hamiltonian(has_spin=False) # first-neighbor hopping
    h.add_haldane(0.15) # Haldane coupling
    h.add_sublattice_imbalance(M) # sublattice imbalance, keeps C3 about a site
    c3 = h.get_chern_mod_n(n=3,center=g.r[0]) # Chern number modulo 3
    c = h.get_chern(nk=30) # Chern number from the Berry curvature
    print("M =",M,"  C mod 3 =",c3,"  C =",round(c,3))
# both give C=1 below the critical imbalance 4.5 t2 = 0.675, and 0 above it

def kane_mele(s):
    """Kane-Mele model with the hopping of the bonds along x equal to s"""
    def fun(r1,r2):
        dr = r1 - r2
        if abs(np.linalg.norm(dr)-1.)>1e-4: return 0.
        return s if abs(dr[1])<1e-4 else 1. # the bond along x
    h = g.get_hamiltonian(fun=fun,has_spin=True)
    h.add_kane_mele(0.1) # spin-orbit coupling
    return h

for s in [1.,1.5,2.5,3.]: # hopping of the bond along x, in units of t
    h = kane_mele(s)
    print("bond =",s,"t   parity Z2 =",h.get_parity_z2())
# the Dirac cones merge at s=2, the gap closes at a time-reversal-invariant
# momentum, and beyond it the strong bonds form a trivial insulator, Z2=0
