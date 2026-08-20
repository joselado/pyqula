# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Exciton band structure of a gapped honeycomb semiconductor.
#
# An exciton is a two-particle state, so besides its binding energy it has
# a dispersion of its own: the bound electron-hole pair can propagate with
# a center-of-mass momentum Q. Solving the Bethe-Salpeter equation at every
# Q along a path gives E_X(Q), the exciton band structure.
#
# Plotted against the electron-hole continuum (the bare transitions of the
# same mean field, obtained by switching the kernel off), the exciton bands
# sit below its edge by their binding energy, and how much they disperse
# says how heavy the exciton is.

import numpy as np
import matplotlib.pyplot as plt
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.add_sublattice_imbalance(1.0) # open a gap, a two-band semiconductor

def coulomb(r1,r2):
    """A Coulomb tail with a soft cutoff at short distance"""
    return 0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25)

W = density_interaction(h,Vr=coulomb) # the electron-hole interaction

nq,n = 21,3 # q-points along the path, and excitons kept at each of them
qvals = np.linspace(-0.5,0.5,nq) # straight through the zone center,
qpath = [[q,0.,0.] for q in qvals] # in reduced units of b1
opts = dict(qpath=qpath,nk=8,n=n,V=W) # same setup for both calculations

# the excitons: the transitions dressed by the electron-hole kernel
qs,es = h.get_exciton_bands(**opts)
# the electron-hole continuum: the bare transitions, no kernel at all
qs0,es0 = h.get_exciton_bands(kernel="none",**opts)

# n excitons at every q-point, so both come back as (nq,n) once reshaped
es = es.reshape((nq,n)).real
es0 = es0.reshape((nq,n)).real

print("lowest exciton and continuum edge along the path")
for iq in range(nq): # loop over the path
    print("  Q = %5.2f   E_X = %.4f   continuum edge = %.4f   binding = %.4f"
            %(qvals[iq],es[iq,0],es0[iq,0],es0[iq,0]-es[iq,0]))

fig = plt.figure(figsize=(6,4))
# the continuum starts at its lowest bare transition and extends upwards,
# so shade everything above that edge up to the top of the axes
top = es0.max()
plt.fill_between(qvals,es0[:,0],top+1.,color="gray",alpha=0.3,
        label="electron-hole continuum")
plt.plot(qvals,es0[:,0],c="gray",lw=1)
plt.plot(qvals,es,c="C3",marker="o",ms=3,lw=1) # the exciton bands
plt.plot([],[],c="C3",marker="o",ms=3,lw=1,label="excitons") # legend entry
plt.ylim(es.min()-0.05,top+0.05)
plt.xlabel("exciton momentum $Q$ (in units of $b_1$)") ; plt.ylabel("$E_X(Q)$")
plt.legend() ; plt.tight_layout()
plt.savefig("exciton_bands.png")
plt.show()
