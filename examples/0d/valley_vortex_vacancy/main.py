# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import islands
from pyqula import spectrum

import numpy as np

# A single vacancy in a honeycomb flake is expected to create a vortex in
# the in-plane valley pseudospin (tau_x,tau_y), since removing one atom is
# the extreme case of an atomically-sharp, intervalley-scattering defect.
#
# The raw in-plane valley expectation value is dominated by the flake's
# own zigzag-edge valley texture (edge states are strongly valley
# polarized), which has nothing to do with the vacancy. To isolate the
# vacancy's own contribution we compute the same quantity on the pristine
# flake and subtract it off, exactly as the vacancy DOS examples in
# examples/embedding/ compare a pristine and a defective system.

g = islands.get_geometry(name="honeycomb",n=8,nedges=6,rot=0.0) # pristine flake
central = g.get_central()[0]
gv = g.remove(central) # flake with a single vacancy

h0 = g.get_hamiltonian(has_spin=False) # pristine
h1 = gv.get_hamiltonian(has_spin=False) # defective

vx0 = spectrum.real_space_vev(h0,operator=h0.get_operator("valley_x"))
vy0 = spectrum.real_space_vev(h0,operator=h0.get_operator("valley_y"))
vx1 = spectrum.real_space_vev(h1,operator=h1.get_operator("valley_x"))
vy1 = spectrum.real_space_vev(h1,operator=h1.get_operator("valley_y"))

# match the defective flake's sites back to the pristine ones by position
# (removing a site keeps the relative order of the remaining ones)
idx_map = np.array([np.argmin((g.x-xi)**2+(g.y-yi)**2)
                     for (xi,yi) in zip(gv.x,gv.y)])
dvx = vx1 - vx0[idx_map] # vacancy-induced part only
dvy = vy1 - vy0[idx_map]

np.savetxt("VALLEY_VORTEX.OUT",np.array([gv.x,gv.y,dvx,dvy]).T)

import matplotlib.pyplot as plt
r = np.sqrt(gv.x**2+gv.y**2)
mask = r<6 # zoom in around the vacancy
fig,ax = plt.subplots(figsize=(7,7))
ax.quiver(gv.x[mask],gv.y[mask],dvx[mask],dvy[mask],
        np.arctan2(dvy[mask],dvx[mask]),cmap="hsv")
ax.scatter([0],[0],c="k",marker="x",s=150,label="vacancy")
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Vacancy-induced in-plane valley (tau_x,tau_y) vortex")
ax.legend()
plt.tight_layout()
plt.show()
