import numpy as np
import style
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian()
qs = np.linspace(-1.,1.,401)
es = np.linspace(-3.,3.,301) # frequencies
chis = [h.get_chi(q=[q,0.,0.],energies=es,nk=401,delta=0.05)[1] for q in qs]
chis = np.array(chis) # shape (len(qs), len(es))

def draw(ax, c):
    m = np.abs(chis.imag).max()
    im = ax.pcolormesh(qs, es, chis.imag.T, cmap=c.diverging,
                       vmin=-m, vmax=m, shading="nearest", rasterized=True)
    style.colorbar(im, ax, r"Im $\chi(q,\omega)$")
    ax.set_xlabel(r"momentum transfer $q$ [$2\pi/a$]")
    ax.set_ylabel(r"frequency $\omega$ [$t$]")

style.save(draw, "charge_map")
