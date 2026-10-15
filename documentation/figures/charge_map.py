import numpy as np
import style
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian()
qs = np.linspace(-1.,1.,40)
chis = [h.get_chi(q=[q,0.,0.],energies=np.linspace(-3,3,100),nk=40,delta=0.1)[1] for q in qs]
es = np.linspace(-3,3,100) # the frequencies of each chi
chis = np.array(chis) # shape (len(qs), len(es))

def draw(ax, c):
    m = np.abs(chis.imag).max()
    im = ax.pcolormesh(qs, es, chis.imag.T, cmap=c.diverging,
                       vmin=-m, vmax=m, shading="nearest", rasterized=True)
    style.colorbar(im, ax, r"Im $\chi(q,\omega)$")
    ax.set_xlabel(r"momentum transfer $q$ [$2\pi/a$]")
    ax.set_ylabel(r"frequency $\omega$ [$t$]")

style.save(draw, "charge_map")
