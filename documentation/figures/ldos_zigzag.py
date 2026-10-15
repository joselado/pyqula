import numpy as np
import style
from pyqula import geometry

g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
(x,y,d) = h.get_ldos(e=0.0,projection="TBRS",nrep=12) # interpolated real-space map, 12 unit cells long
(x2,y2,es,ldos) = h.get_multildos(energies=np.linspace(-2.0,2.0,401),
                                  nk=1000,delta=0.05) # one map per energy, one value per site

d = d.reshape(600,600) # TBRS gives a 600 x 600 grid, x first
iy = np.argsort(y2) # sites ordered across the ribbon

def draw(ax, c):
    # the position across the ribbon is the vertical axis of both panels
    im = ax[0].pcolormesh(np.unique(x), np.unique(y), d.T, cmap=c.cmap, shading="auto")
    style.colorbar(im, ax[0], r"LDOS at $\omega=0$")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("along the ribbon $x$ [$a$]")
    ax[0].set_ylabel("across the ribbon $y$ [$a$]")
    im = ax[1].pcolormesh(es, y2[iy], ldos[:,iy].T, cmap=c.cmap, shading="nearest", vmax=1.2)
    style.colorbar(im, ax[1], r"LDOS $D(y,\omega)$")
    ax[1].set_xlabel(r"energy $\omega$ [$t$]")
    ax[1].set_ylabel("across the ribbon $y$ [$a$]")

style.save(draw, "ldos_zigzag", npanels=2, gridspec_kw={"width_ratios":[1,1.6]})
