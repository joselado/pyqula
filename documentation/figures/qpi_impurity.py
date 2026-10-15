import numpy as np
import style
from pyqula import geometry

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
ns = 69 # a multiple of three, so the K points are on the q mesh
r,ldos_r,q,qpi_q = h.get_qpi_impurity(nsuper=ns,
        impurities=[{"position": [0.,0.,0.], "onsite": 3.0}],
        energies=0.3,num_waves=2*ns**2, # all the states, a full diagonalization
        nk=1,delta=0.1)

ldos_r, qpi_q = np.ravel(ldos_r), np.ravel(qpi_q) # one energy
r0 = r[np.argmin(ldos_r)] # the impurity site, where the density drops
q0 = np.argmin(np.linalg.norm(q,axis=1)) # q=0 holds the mean LDOS
qpi_q[q0] = np.nan # leave it out of the color scale
ok = ~np.isnan(qpi_q)
K = 4*np.pi/(3*np.sqrt(3)) # |K| for a first-neighbor distance a=1

def draw(ax, c):
    dl = ldos_r - ldos_r.mean() # the change around the mean
    w = np.linalg.norm(r-r0,axis=1) < 12 # a window around the impurity
    m = np.percentile(np.abs(dl[w]),95) # the impurity site itself saturates
    sc = ax[0].scatter(r[w,0]-r0[0], r[w,1]-r0[1], c=dl[w], s=16,
            cmap=c.diverging, vmin=-m, vmax=m)
    style.colorbar(sc, ax[0], r"$D(\mathbf{r},\omega)-\bar D$")
    ax[0].set_xlabel("$x$ [$a$]")
    ax[0].set_ylabel("$y$ [$a$]")
    im = ax[1].tripcolor(q[ok,0], q[ok,1], qpi_q[ok], cmap=c.cmap,
            shading="gouraud", vmax=np.percentile(qpi_q[ok],99))
    style.colorbar(im, ax[1], r"QPI$(\mathbf{q},\omega)$")
    for (x,y,s) in [(0.55,0,r"$\Gamma$"),(0,K,"K"),(0,-K,"K'")]: # K and K'
        ax[1].text(x, y, s, color="w", ha="center", va="center") # in the dark center of their ring
    ax[1].set_xlabel("$q_x$ [$1/a$]")
    ax[1].set_ylabel("$q_y$ [$1/a$]")
    for a in ax: a.set_aspect("equal", adjustable="datalim")

style.save(draw, "qpi_impurity", npanels=2)
