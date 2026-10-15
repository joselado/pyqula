import numpy as np
import style
from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
h.add_swave(0.2) # add pairing, doubling the basis into Nambu space
electron = h.get_operator("electron") # electron component
hole = h.get_operator("hole") # hole component
(k,e,c) = h.get_bands(operator=electron,nk=1500) # electron weight of each state

def draw(ax, cl):
    sc = ax.scatter(k, e, c=c, s=3, cmap=cl.diverging, vmin=0, vmax=1)
    style.colorbar(sc, ax, r"electron weight $|u|^2$")
    style.kpath(ax, k, [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"], at=[0,1/3,1/2,2/3,1])
    ax.set_xlabel("momentum along the k-path")
    ax.set_ylabel("energy [$t$]")
    ax.set_ylim(-3.5, 3.5)

style.save(draw, "operators_nambu")
