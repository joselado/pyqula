import numpy as np
import style
from pyqula import geometry

g = geometry.square_lattice()
h = g.get_hamiltonian(has_spin=False)
(qs,es,qpi) = h.get_qpi(mode="tmatrix",impurity=5.0,nk=120,nsuper=2,
        delta=1e-1,energies=[-1.0]) # a strong point impurity
(qs,es,qpib) = h.get_qpi(mode="born",impurity=5.0,nk=120,nsuper=2,
        delta=1e-1,energies=[-1.0]) # its Born approximation

cut = np.abs(qs[:,1])<1e-9 # the line q=(q_x,0)
order = np.argsort(qs[cut,0])

def draw(ax, c):
    im = ax[0].tripcolor(qs[:,0], qs[:,1], qpi[0], cmap=c.cmap, shading="gouraud",
            vmax=np.percentile(qpi[0],99.5))
    style.colorbar(im, ax[0], r"$|\delta\rho(\mathbf{q},\omega)|$")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$q_x$ [$2\pi/a$]")
    ax[0].set_ylabel(r"$q_y$ [$2\pi/a$]")
    for (y,col,lab) in [(qpi[0],c.blue,"T-matrix"),(qpib[0],c.orange,"Born")]:
        y = y[cut][order]
        ax[1].plot(qs[cut,0][order], y/np.max(y), color=col, label=lab)
    ax[1].set_xlabel(r"$q_x$ [$2\pi/a$], at $q_y=0$")
    ax[1].set_ylabel(r"$|\delta\rho(\mathbf{q},\omega)|$, normalized")
    ax[1].legend()

style.save(draw, "qpi_tmatrix", npanels=2)
