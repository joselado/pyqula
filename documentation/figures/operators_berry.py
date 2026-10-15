import numpy as np
import style
from pyqula import geometry
g = geometry.honeycomb_lattice() # honeycomb lattice
h = g.get_hamiltonian(has_spin=False) # spinless Hamiltonian
h.add_haldane(0.1) # Haldane coupling, a Chern insulator
(k,e,b) = h.get_bands(operator="berry") # bands, and Berry curvature of each state
bmax = np.abs(b).max()

def draw(ax, c):
    i = np.argsort(np.abs(b)) # draw the largest curvature on top
    sc = ax.scatter(k[i], e[i], c=b[i], s=6, cmap=c.diverging, vmin=-bmax, vmax=bmax)
    style.colorbar(sc, ax, r"Berry curvature $\Omega$")
    ax.set_xticks(np.array([0,1/3,1/2,2/3,1])*k.max(),
                  [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"])
    ax.set_xlim(0, k.max())
    ax.set_xlabel("momentum along the k-path")
    ax.set_ylabel("energy [$t$]")

style.save(draw, "operators_berry")
