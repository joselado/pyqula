import numpy as np
import style
from pyqula import geometry
from pyqula import supercell
from pyqula import latticegas
np.random.seed(1) # fixed seed, so that every run gives the same figure

g = geometry.triangular_lattice()
g = supercell.turn_orthorhombic(g)
g = g.get_supercell(10) # periodic, so no site sits at an edge

lg = latticegas.LatticeGas(g,filling=1./3.) # 1/3 of the sites randomly occupied
lg.add_interaction(Jij=[1.,1.,1.]) # first, second and third neighbor repulsion
es = lg.optimize_energy(temp=0.5,ntries=1e4) # simulated annealing
(q,sq) = lg.get_structure_factor(nq=81) # structure factor on a grid of q

def draw(ax, c):
    occ = lg.den > 0.5 # occupied sites
    ax[0].scatter(g.r[:,0], g.r[:,1], s=10, color=c.ink2, alpha=0.4)
    ax[0].scatter(g.r[occ,0], g.r[occ,1], s=45, color=c.orange)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("$x$") ; ax[0].set_ylabel("$y$")
    ax[1].plot(es/len(g.r), color=c.blue)
    ax[1].set_xlabel("Monte Carlo step")
    ax[1].set_ylabel("energy per site [$J$]")
    n = int(np.sqrt(len(sq)))
    im = ax[2].pcolormesh(q[:,0].reshape(n,n), q[:,1].reshape(n,n),
                          sq.reshape(n,n), cmap=c.cmap, shading="auto")
    style.colorbar(im, ax[2], r"structure factor $S(\mathbf{q})$")
    ax[2].set_aspect("equal")
    ax[2].set_xlabel("$q_x$") ; ax[2].set_ylabel("$q_y$")

style.save(draw, "latticegas_order", npanels=3)
