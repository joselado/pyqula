# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Optical absorption and second-harmonic generation with excitons on a fine
# k-mesh. The sums over excitons are done with a Chebyshev expansion of the
# Bethe-Salpeter matrix at the complex energy w + i delta (method="kpm"),
# without diagonalizing it. The result is the same as with method="dense",
# and the cost grows only linearly with the number of k-points. Above
# the gap the curves oscillate, because the mesh does not resolve the
# broadening there: the mesh spacing dk has to satisfy delta >= v dk/2,
# with v the band velocity. Going from a 30x30 to a 60x60 mesh makes the
# oscillations of the absorption about three times smaller. The second
# harmonic above the gap needs a finer mesh still at this delta, since it
# depends on how the pairs change with k. The exciton
# at 1.845t, below the gap, and its second-harmonic peak at half that
# energy are at the same place on both meshes.

import numpy as np
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False) # spinless, one spin species
h.add_sublattice_imbalance(1.0) # gap 2t at K and K'
W = density_interaction(h,Vr=lambda r1,r2: 0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25))

es = np.linspace(0.,2.6,131)
kw = dict(energies=es,delta=0.04,V=W,method="kpm")
(ws,s30) = h.get_excitonic_optical_conductivity(nk=30,component="xx",**kw)
(ws,s60) = h.get_excitonic_optical_conductivity(nk=60,component="xx",**kw)
(ws,chi30) = h.get_excitonic_shg_susceptibility(nk=30,components="xxx",**kw)
(ws,chi60) = h.get_excitonic_shg_susceptibility(nk=60,components="xxx",**kw)
print("absorption maximum at ",ws[np.argmax(s60.real)]) # the exciton
print("SHG maximum at        ",ws[np.argmax(np.abs(chi60))]) # half of it

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(ws,2*np.pi*s30.real,c="gray",label="30x30")
plt.plot(ws,2*np.pi*s60.real,c="blue",label="60x60")
plt.xlabel("Frequency [t]") ; plt.ylabel("Re $\\sigma_{xx}$ [$e^2/h$]")
plt.legend()
plt.subplot(1,2,2)
plt.plot(ws,np.abs(chi30),c="gray",label="30x30")
plt.plot(ws,np.abs(chi60),c="blue",label="60x60")
plt.xlabel("Frequency [t]") ; plt.ylabel("$|\\chi_{xxx}|$ [nm$^2$/V]")
plt.legend()
plt.tight_layout()
plt.show()
