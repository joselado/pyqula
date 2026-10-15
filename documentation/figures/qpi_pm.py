import numpy as np
import style
from pyqula import geometry

g = geometry.triangular_lattice()
h = g.get_hamiltonian(has_spin=False)
(qs,es,qpi) = h.get_qpi(mode="pm",nk=50,delta=1e-1,energies=np.linspace(-6.,6.,100)) # QPI at each energy

def draw(ax, c):
    for a, e0 in zip(ax, [5.0, -2.5]): # near the top and near the bottom of the band
        i = np.argmin(np.abs(es-e0)) # the energy closest to e0
        im = a.tripcolor(qs[:,0], qs[:,1], qpi[i], cmap=c.cmap, shading="gouraud",
                vmax=np.percentile(qpi[i],99)) # the q=0 peak saturates
        style.colorbar(im, a, r"QPI $P(\mathbf{q},\omega)$")
        a.set_title(r"$\omega=%.2f\,t$" % es[i])
        a.set_aspect("equal")
        a.set_xlabel(r"$q_x$ [$2\pi/a$]")
        a.set_ylabel(r"$q_y$ [$2\pi/a$]")

style.save(draw, "qpi_pm", npanels=2)
