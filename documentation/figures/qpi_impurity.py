import numpy as np
import style
from pyqula import geometry

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
r,ldos_r,q,qpi_q = h.get_qpi_impurity(nsuper=12,
        impurities=[{"position": [0.,0.,0.], "onsite": 3.0}],
        energies=0.3,num_waves=60,nk=2,delta=0.2)

ldos_r, qpi_q = np.ravel(ldos_r), np.ravel(qpi_q) # one energy
q0 = np.argmin(np.linalg.norm(q,axis=1)) # q=0 holds the mean LDOS
qpi_q[q0] = np.nan # leave it out of the color scale

def draw(ax, c):
    dl = ldos_r - ldos_r.mean() # the change around the mean
    m = np.percentile(np.abs(dl),98) # the impurity site itself saturates
    sc = ax[0].scatter(r[:,0], r[:,1], c=dl, s=14, cmap=c.diverging, vmin=-m, vmax=m)
    style.colorbar(sc, ax[0], r"$D(\mathbf{r},\omega)-\bar D$")
    ax[0].set_xlabel("$x$ [$a$]")
    ax[0].set_ylabel("$y$ [$a$]")
    sc = ax[1].scatter(q[:,0], q[:,1], c=qpi_q, s=80, marker="h", cmap=c.cmap)
    style.colorbar(sc, ax[1], r"QPI$(\mathbf{q},\omega)$")
    ax[1].set_xlabel("$q_x$ [$1/a$]")
    ax[1].set_ylabel("$q_y$ [$1/a$]")
    for a in ax: a.set_aspect("equal")

style.save(draw, "qpi_impurity", npanels=2)
