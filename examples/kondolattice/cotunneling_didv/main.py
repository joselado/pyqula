# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Tunneling spectrum (STM dI/dV) of the Kondo lattice mean field.
# The tip tunnels into the conduction electron with amplitude tc, and into
# the composite f fermion (through a spin flip of the local moment) with
# amplitude tf. The two paths interfere, so that
#   dI/dV = tc^2 N_cc + tf^2 N_ff + 2 tc tf N_cf
# at the site below the tip (Maltseva, Dzero and Coleman, arXiv:0910.1138;
# Figgins and Morr, arXiv:1001.4530). Energies are measured from the chemical
# potential of the self-consistent loop.
#
# Here we compare with what the first paper says about the lattice:
#  1) the hybridization gap between the two peaks is 2 V^2/D for a
#     symmetric band of half width D, and V^2 (1/d1 + 1/d2) in general,
#     with d1 and d2 the distances from the f level lam to the band edges;
#  2) the line shape is G_psi = (tc + tf V/(w-lam))^2 G_cc + tf^2/(w-lam),
#     where G_cc is the conduction Green function of the lattice.
# The equation of the paper for each k prints the first term only. The last
# one is the bare f level. It removes the pole of the first term at w = lam,
# and the formula of the paper summed over k has it.

import numpy as np
from scipy.signal import find_peaks
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

gc = geometry.chain()
hc = gc.get_hamiltonian(has_spin=True) # conduction electrons, t=1 chain
h = KondoLatticeHamiltonian(hc)
J = 1.5
h2 = h.get_mean_field_hamiltonian(J=J, filling=0.15, nk=200,
        mf=([0.3+0j], [0.0]), mix=0.3, maxerror=1e-7, maxite=3000)
V = h2.hybridization[0].real
# onsite energies of the converged Hamiltonian, measured from the chemical
# potential: the conduction band center and the f level
e0 = h2.intra[0, 0].real
lam = h2.intra[2, 2].real

# 1) the gap, from the spectrum with tf=0 (the pure conduction LDOS)
nk = 4000
(e, d) = h2.get_cotunneling_didv(tc=1.0, tf=0.0, nk=20000, delta=2e-4,
        emax=0.4, ne=8001)
dos = d[0]
# the two sharp peaks of the spectrum are the band edges of the gap
peaks, _ = find_peaks(dos, height=0.2*dos.max())
top2 = np.sort(peaks[np.argsort(dos[peaks])[-2:]])
low, high = e[top2[0]], e[top2[1]]
d1 = lam-(e0-2.0)  # distance from lam to the bottom of the band
d2 = (e0+2.0)-lam  # distance from lam to the top of the band
print("hybridization V, f level lam, band edges d1 and d2")
print(V, lam, d1, d2)
print("gap from dI/dV (tf=0), distance between its two peaks")
print(high-low)
print("gap from V^2 (1/d1+1/d2), perturbative in V/d")
print(V**2*(1/d1+1/d2))
# exact indirect gap of the two-band spectrum
wm = (e0+2.0+lam-np.sqrt((e0+2.0-lam)**2+4*V**2))/2.
wp = (e0-2.0+lam+np.sqrt((e0-2.0-lam)**2+4*V**2))/2.
print("exact indirect gap w+(bottom)-w-(top)")
print(wp-wm)

# 2) the line shape with the interference, against the formula
tf = 0.3
(e, d, parts) = h2.get_cotunneling_didv(tc=1.0, tf=tf, nk=nk, delta=1e-2,
        emax=1.5, ne=601, return_parts=True)
w = e+1j*1e-2
ks = np.linspace(0, 2*np.pi, nk, endpoint=False)
G = np.mean([(1+tf*V/(w-lam))**2/(w-e0-2*np.cos(k)-V**2/(w-lam))
        for k in ks], axis=0) + tf**2/(w-lam)
ref = -2*G.imag/np.pi # two spin components
print("largest difference to the formula, relative to the maximum")
print(np.max(np.abs(d[0]-ref))/np.max(ref))
# without the bare f pole the paper formula differs by this amount
G0 = G-tf**2/(w-lam)
print("same without the f pole term")
print(np.max(np.abs(d[0]+2*G0.imag/np.pi))/np.max(ref))

# the three parts, to see the interference. Flipping the sign of tf
# flips the interference and keeps the conduction and f parts
(e, dm) = h2.get_cotunneling_didv(tc=1.0, tf=-tf, nk=nk, delta=1e-2,
        emax=1.5, ne=601)
asym = np.sum(parts["cf"][0])/np.sum(np.abs(parts["cf"][0]))
print("net weight of the interference term, in [-1,1]")
print(asym)
print("dI/dV(tf)+dI/dV(-tf) = 2 (cc+ff), difference:")
print(np.max(np.abs(d[0]+dm[0]-2*(parts["cc"][0]+parts["ff"][0]))))

# thermal broadening of the spectrum, T in the units of the hopping
(e, dT) = h2.get_cotunneling_didv(tc=1.0, tf=tf, nk=nk, delta=1e-2,
        T=0.05, emax=1.5, ne=601)
np.savetxt("DIDV.OUT", np.array([e, d[0], dT[0], parts["cc"][0],
        parts["ff"][0], parts["cf"][0]]).T)
