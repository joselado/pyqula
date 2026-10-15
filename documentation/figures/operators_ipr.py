import numpy as np
from matplotlib.colors import LogNorm
import style
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
h.add_onsite(0.3) # add a uniform onsite energy
ipr = h.get_operator("IPR") # IPR operator
(k,e,p) = h.get_bands(operator=ipr) # bands, and IPR of each state

def draw(ax, c):
    i = np.argsort(p) # draw the localized states on top
    sc = ax.scatter(k[i], e[i], c=p[i], s=4, cmap=c.cmap,
                    norm=LogNorm(vmin=1e-2, vmax=1)) # logarithmic color scale
    style.colorbar(sc, ax, "inverse participation ratio")
    style.kpath(ax, k, ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel("momentum along the ribbon")
    ax.set_ylabel("energy [$t$]")
    ax.set_ylim(-1.2, 1.8)

style.save(draw, "operators_ipr")
