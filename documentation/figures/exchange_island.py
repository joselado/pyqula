import numpy as np
import style
from pyqula import geometry
np.random.seed(1) # the same random guess in every run
g = geometry.honeycomb_lattice().get_supercell(12) # a honeycomb island
g.dimensionality = 0 # make it finite
h = g.get_hamiltonian(has_spin=True,is_sparse=True) # keep the matrices sparse
h.add_rashba(0.2) # spin-orbit coupling, so the moments are not collinear
h = h.get_mean_field_hamiltonian(U=3.0,filling=0.5,mf="random",
                                  integration="kpm",npol=150,mix=0.5,
                                  maxerror=1e-3)
m = np.linalg.norm(h.get_magnetization(),axis=1) # length of the moment of each site

def draw(ax, c):
    sc = ax.scatter(g.r[:,0], g.r[:,1], c=m, s=22, cmap=c.cmap, vmin=0.2, vmax=0.85)
    style.colorbar(sc, ax, r"length of the moment $|\vec m|$")
    ax.set_aspect("equal")
    ax.axis("off")

style.save(draw, "exchange_island")
