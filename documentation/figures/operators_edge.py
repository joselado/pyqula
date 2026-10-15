import numpy as np
import style
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
b = h.get_operator("bulk") # bulk operator
e = h.get_operator("edge") # edge operator
(k,en,w) = h.get_bands(operator=e) # bands, and edge weight of each state

def draw(ax, c):
    i = np.argsort(w) # draw the edge states on top
    sc = ax.scatter(k[i], en[i], c=w[i], s=4, cmap=c.cmap, vmin=0, vmax=1)
    style.colorbar(sc, ax, r"weight on the edge $\langle \hat E\rangle$")
    style.kpath(ax, k, ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel("momentum along the ribbon")
    ax.set_ylabel("energy [$t$]")
    ax.set_ylim(-1.5, 1.5)

style.save(draw, "operators_edge")
