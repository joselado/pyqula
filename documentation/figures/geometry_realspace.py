import numpy as np
import style
from pyqula import islands
from pyqula import topology
g = islands.get_geometry(name="honeycomb",n=8,nedges=3) # a honeycomb island
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
(r,c) = topology.real_space_chern(h) # spatially-resolved Berry curvature
r = np.array(r) # positions of the sites

def draw(ax, cl):
    sc = ax.scatter(r[:,0], r[:,1], c=c, s=22, cmap=cl.diverging, vmin=-3, vmax=3)
    style.colorbar(sc, ax, r"local Chern marker $C(\mathbf{r})$")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [$d$]")
    ax.set_ylabel("$y$ [$d$]")

style.save(draw, "geometry_realspace")
