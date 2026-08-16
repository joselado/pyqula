# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


# Superfluid weight of a self-consistent attractive-Hubbard superconductor
#
# The superfluid weight (superfluid stiffness) is the rigidity of the grand
# potential against a phase twist of the order parameter,
#     D_s^ab = (1/V) d^2 Omega / dQ_a dQ_b
# with |Delta| held at its self-consistent value.  It is what actually makes
# a superconductor a superconductor: a finite pairing amplitude with zero
# stiffness carries no supercurrent.  In a multiband system it splits into a
# conventional part, built from the band velocities, and a quantum-geometric
# part, built from the interband matrix elements of the current operator and
# related to the quantum metric of the normal-state bands (Peotta & Toermae,
# Nat. Commun. 6, 8944 (2015); Liang et al., PRB 95, 024515 (2017)).
#
# Here we sweep the attractive interaction of a honeycomb-lattice Hubbard
# model, solve the BdG mean field at each U, and plot the gap, both parts of
# the superfluid weight and the BKT temperature T_BKT = (pi/8) D_s(T_BKT).

import glob
import numpy as np
from pyqula import geometry

g = geometry.honeycomb_lattice()

Us = np.linspace(1.0, 3.0, 9)   # attraction strengths
deltas, conv, geom, tbkt = [], [], [], []

for U in Us:
    for name in glob.glob("*.pkl"): os.remove(name)  # no stale SCF guess
    h = g.get_hamiltonian()
    h.turn_nambu()               # BdG Hamiltonian, no pairing yet
    h = h.get_mean_field_hamiltonian(U=-U, filling=0.2, mf="swave",
                                     nk=10, mix=0.8, maxerror=1e-6)
    if h is None:                # the SCF returns None if it did not converge
        print("U =", U, "did not converge") ; continue
    delta = np.abs(np.mean(h.extract("swave")))
    # the tensor, split into its conventional and geometric contributions
    out = h.get_superfluid_weight(nk=20, decompose=True)
    tb = h.get_bkt_temperature(nk=16)
    deltas.append(delta)
    conv.append(out["conventional"][0, 0])
    geom.append(out["geometric"][0, 0])
    tbkt.append(tb)
    print("U = %5.2f   Delta = %7.4f   D_conv = %7.4f   D_geom = %7.4f"
          "   T_BKT = %7.4f" % (U, delta, conv[-1], geom[-1], tb))

conv = np.array(conv) ; geom = np.array(geom)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(deltas, conv+geom, "o-", label="total $D_s$")
ax[0].plot(deltas, conv, "s--", label="conventional")
ax[0].plot(deltas, geom, "^--", label="geometric")
ax[0].set_xlabel(r"$|\Delta|$") ; ax[0].set_ylabel(r"$D_s^{xx}$")
ax[0].legend() ; ax[0].set_title("superfluid weight, honeycomb Hubbard")

ax[1].plot(deltas, tbkt, "o-")
ax[1].set_xlabel(r"$|\Delta|$") ; ax[1].set_ylabel(r"$T_{BKT}$")
ax[1].set_title(r"$T_{BKT} = \frac{\pi}{8} D_s(T_{BKT})$")

plt.tight_layout()
plt.show()
