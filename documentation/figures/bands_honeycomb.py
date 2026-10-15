import numpy as np
import style
from pyqula import geometry

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
(k,e,c) = h.get_bands(operator="velocity") # bands colored by group velocity

def draw(ax, c_):
    ax.scatter(k, e, s=10, color=c_.ink2) # outline, so that zero velocity stays visible
    sc = ax.scatter(k, e, c=c, s=6, cmap=c_.cmap, vmin=0)
    style.colorbar(sc, ax, r"group velocity [$ta/\hbar$]")
    style.kpath(ax, k, [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"], at=[0,1/3,1/2,2/3,1])
    ax.set_xlabel("momentum along the k-path")
    ax.set_ylabel("energy [$t$]")

style.save(draw, "bands_honeycomb")
