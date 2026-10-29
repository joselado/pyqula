# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Exciton bands of a gapped honeycomb semiconductor as a spectral function.
#
# The exciton density of states A(Q,w) along a q-path shows the exciton
# bands as bright lines, the bound exciton below the electron-hole
# continuum. Projected on sublattice A, acting on the electron, it gives
# the probability that the electron of the exciton sits on A. The sublattice
# imbalance puts the conduction band on A and the valence band on B, so the
# electron is mostly on A and the hole mostly on B.

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

ws = np.linspace(1.7,3.0,200) # energy grid
opts = dict(V=W,nq=20,nk=18,energies=ws,delta=0.05)
qs,es,ds = h.get_exciton_kdos_bands(**opts) # exciton density of states
qs,es,da = h.get_exciton_kdos_bands(operator="sublatticeA",**opts) # electron on A

nq = qs.max()+1
D = ds.reshape(nq,-1).T # (energy, q-point)
DA = da.reshape(nq,-1).T # the part of it with the electron on A

i = np.argmin(np.abs(ws-1.86)) # the bottom of the exciton band
print("probability of the electron on A at 1.86t, at each q-point")
print(np.round(DA[i]/D[i],3))

fig,ax = plt.subplots(1,2,figsize=(9,4),sharey=True)
m0 = ax[0].pcolormesh(np.arange(nq),ws,np.sqrt(D),shading="nearest")
ax[0].set_title("exciton DOS (square root)")
m1 = ax[1].pcolormesh(np.arange(nq),ws,np.sqrt(DA),shading="nearest")
ax[1].set_title("electron on sublattice A (square root)")
for a in ax: a.set_xlabel("q-point along the path")
ax[0].set_ylabel("energy [t]")
plt.tight_layout()
plt.savefig("exciton_kdos_bands.png")
plt.show()
