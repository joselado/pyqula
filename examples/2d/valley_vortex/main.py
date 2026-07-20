# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import spectrum

import numpy as np

# A single vacancy in a honeycomb flake is expected to create a vortex in
# the in-plane valley pseudospin (tau_x,tau_y), since removing one atom is
# the extreme case of an atomically-sharp, intervalley-scattering defect.
#

g = geometry.honeycomb_lattice()
g = g.get_supercell(3)
central = g.get_central()[0]
gv = g.remove(central) # flake with a single vacancy

hv = gv.get_hamiltonian(has_spin=False) # defective

dvx = spectrum.real_space_vev(hv,operator=hv.get_operator("valley_x"),nk=1)
dvy = spectrum.real_space_vev(hv,operator=hv.get_operator("valley_y"),nk=1)


np.savetxt("VALLEY_VORTEX.OUT",np.array([gv.x,gv.y,dvx,dvy]).T)

import matplotlib.pyplot as plt
r = np.sqrt(gv.x**2+gv.y**2)
mask = r<6 # zoom in around the vacancy
fig,ax = plt.subplots(figsize=(7,7))
ax.quiver(gv.x[mask],gv.y[mask],dvx[mask],dvy[mask],
        np.arctan2(dvy[mask],dvx[mask]),cmap="hsv")
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Vacancy-induced in-plane valley (tau_x,tau_y) vortex")
ax.legend()
plt.tight_layout()
plt.show()
