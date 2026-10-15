import numpy as np
import style
from pyqula import specialgeometry
from pyqula.graphenetk.geometry import GrapheneGeometry
from pyqula.graphenetk.hamiltonian import GrapheneHamiltonian

g0 = specialgeometry.twisted_bilayer(m0=15) # ~2 degree twist
g = GrapheneGeometry(g0).relax() # AA area shrinks, AB/BA domains grow
h = GrapheneHamiltonian(g)
(k,e) = h.get_bands(num_bands=20,nk=40) # 40 k-points, to keep it short
h0 = GrapheneHamiltonian(GrapheneGeometry(g0)) # the rigid lattice, as a reference
(k0,e0) = h0.get_bands(num_bands=20,nk=40)

u = g.r - g0.r # in-plane displacement of each atom
top = g0.r[:,2] > 0 # atoms of the top layer

def draw(ax, c):
    cells = [(i,j) for i in range(-2,3) for j in range(-2,3)] # 5x5 moire cells
    r = np.concatenate([g0.r[top] + i*g0.a1 + j*g0.a2 for (i,j) in cells])
    du = np.concatenate([u[top] for (i,j) in cells])
    umod = np.sqrt(du[:,0]**2 + du[:,1]**2)
    sc = ax[0].scatter(r[:,0], r[:,1], c=umod, s=10, cmap=c.cmap, vmin=0)
    style.colorbar(sc, ax[0], "displacement $|\\mathbf{u}|$")
    q = np.unique(np.round(r[:,:2]/5.), axis=0, return_index=True)[1] # one arrow per 5x5 area
    ax[0].quiver(r[q,0], r[q,1], du[q,0], du[q,1], color="black", scale=0.005,
                 scale_units="xy", width=0.005, headwidth=4)
    L = 1.1*np.linalg.norm(g0.a1)
    ax[0].set_xlim(-L, L) ; ax[0].set_ylim(-L, L) ; ax[0].set_aspect("equal")
    ax[0].set_xlabel("$x$") ; ax[0].set_ylabel("$y$")
    ax[1].scatter(k0, e0, s=8, color=c.ink2, label="rigid")
    ax[1].scatter(k, e, s=8, color=c.orange, label="relaxed")
    ticks = np.array([0, 1/3, 1/2, 2/3, 1])*max(k) # the path goes through K, M and K'
    ax[1].set_xticks(ticks, [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"])
    ax[1].set_xlim(0, max(k))
    ax[1].set_ylabel("energy [$t$]")
    ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                 markerscale=2)

style.save(draw, "relaxation_twisted", npanels=2)
