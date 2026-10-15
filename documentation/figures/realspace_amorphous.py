import style
from pyqula import geometry
from pyqula import topology
from pyqula.geometrytk import amorphous
from pyqula.topologytk.realspace import site_areas
import numpy as np
g = geometry.amorphous_lattice(L=36.,density=0.6,seed=0) # 778 sites at random positions
h = amorphous.amorphous_chern_hamiltonian(g,M=-0.5) # Agarwala-Shenoy model, half filling
(r,c) = topology.real_space_chern(h) # local marker, per site
A = site_areas(g) # area of the Voronoi cell of each site
bulk = np.max(np.abs(g.r[:,0:2]),axis=1)<6. # central square, far from the edges
C = np.sum(c[bulk]*A[bulk])/np.sum(A[bulk]) # area-weighted average of the marker

def draw(ax, cl):
    sc = ax.scatter(g.r[:,0], g.r[:,1], c=c, s=14, cmap=cl.diverging, vmin=-3, vmax=3)
    style.colorbar(sc, ax, r"local Chern marker $C(\mathbf{r})$")
    ax.plot([-6,6,6,-6,-6], [-6,-6,6,6,-6], color="gray", lw=1.5) # the central square
    ax.set_aspect("equal")
    ax.set_xlabel("position $x$")
    ax.set_ylabel("position $y$")
    ax.set_title(f"C = {C:.2f} in the central square")

style.save(draw, "realspace_amorphous")
