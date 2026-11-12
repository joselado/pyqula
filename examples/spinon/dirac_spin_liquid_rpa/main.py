# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Dirac spin liquid of the J1-J2 Heisenberg model in the triangular lattice,
# and its dynamical spin response, following arXiv:2503.13831.
#
# The spins are written with pseudofermions, S_i = 1/2 f_i^dagger sigma f_i,
# with one fermion per site. We keep only the bond channel of the exchange
# (decoupling="bond"), so the spinons hop with t_ij = J_ij chi_ij / 2 and
# there is no magnetic mean field. The hopping pattern is held with ansatz:
# here the pi-flux pattern, with flux pi in the up triangles and zero in the
# down ones, whose spinons have Dirac cones.
#
# The exchange between the spinons comes back in the spin response, in the
# random phase approximation, chi = chi0 [1 + J(q) chi0]^-1. It gives a
# sharp mode below the two-spinon continuum. The mode goes to zero energy
# at K for J2 = 0.0716 J1 (120 degree order) and at M for J2 = 0.176 J1
# (stripe order). Between the two the liquid is stable.

import numpy as np
from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian

g0 = geometry.triangular_lattice()
g = g0.get_supercell((2,1)) # two sites, the cell of the pi-flux pattern
a = g.get_hamiltonian(has_spin=True) # first-neighbor hoppings, all +1
ansatz = dict()
for (d,m) in a.get_dict().items(): # put the signs of the pi-flux pattern
    m = np.array(m)
    for i in range(2):
        for j in range(2):
            dr = g.r[j] + d[0]*g.a1 + d[1]*g.a2 - g.r[i] # the bond
            if abs(dr[1])<1e-6: continue # a bond along a1 keeps +1
            p = i if dr[1]>0 else i + int(round(dr[0]-0.5)) # its lower end
            m[2*i:2*i+2,2*j:2*j+2] *= (-1)**p
    ansatz[d] = m

K = [2/3,1/3,0.] # K point of the lattice, in the vectors of the supercell
M = [0.,1/2,0.] # M point of the lattice

# the stiffness is the smallest eigenvalue of 1 + J(q) chi0(q,0): it is
# positive when the liquid is stable against magnetic order at q
print("J2/J1, stiffness at K, stiffness at M")
for J2 in [0.0,0.0716,0.125,0.176,0.25]:
    h = SpinonHamiltonian(g) # no bare hopping, a pure spin model
    # the small temperature keeps the two sites equivalent on the k-mesh
    h2 = h.get_mean_field_hamiltonian(J1=1.0,J2=J2,decoupling="bond",
            ansatz=ansatz,nk=20,T=0.01)
    # an odd nk keeps the Dirac points off the k-mesh
    rK = h2.get_rpa_stiffness(K,nk=31,delta=1e-3,T=1e-3)
    rM = h2.get_rpa_stiffness(M,nk=31,delta=1e-3,T=1e-3)
    print("%.4f  %8.4f  %8.4f" % (J2,rK,rM))

# spin response at K in the stable liquid, with the factor 2 on the exchange
# that accounts for the Gutzwiller projection in an approximate way
h = SpinonHamiltonian(g)
h2 = h.get_mean_field_hamiltonian(J1=1.0,J2=0.125,decoupling="bond",
        ansatz=ansatz,gutzwiller=2.0,nk=20,T=0.02)
energies = np.linspace(0.,3.,100)
(es,chi0) = h2.get_spinchi_full(q=K,energies=energies,delta=0.05,nk=31,RPA=False)
(es,chi) = h2.get_spinchi_full(q=K,energies=energies,delta=0.05,nk=31)
# sum over the three spin components and the two sites
s0 = np.array([np.trace(c).imag for c in chi0]) # two-spinon continuum
s = np.array([np.trace(c).imag for c in chi]) # with the sharp mode
print("Peak of the free spinon response at energy",es[np.argmax(s0)])
print("Peak of the RPA response at energy",es[np.argmax(s)])
np.savetxt("SPIN_RESPONSE.OUT",np.array([es,s0,s]).T)
