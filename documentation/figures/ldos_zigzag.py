import numpy as np
import style
from pyqula import geometry

g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
(x,y,d) = h.get_ldos(e=0.0,projection="TBRS") # interpolated real-space map
(x2,y2,es,ldos) = h.get_multildos(energies=np.linspace(-2.0,2.0,100)) # one map per energy, one value per site

d = d.reshape(600,600) # TBRS gives a 600 x 600 grid, x first
iy = np.argsort(y2) # sites ordered across the ribbon

def draw(ax, c):
    im = ax[0].pcolormesh(np.unique(y), np.unique(x), d, cmap=c.cmap, shading="auto")
    style.colorbar(im, ax[0], r"LDOS at $\omega=0$")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("position across the ribbon $y$ [$a$]")
    ax[0].set_ylabel("position along the ribbon $x$ [$a$]")
    im = ax[1].pcolormesh(y2[iy], es, ldos[:,iy], cmap=c.cmap, shading="auto", vmax=1.2)
    style.colorbar(im, ax[1], r"LDOS $D(y,\omega)$")
    ax[1].set_xlabel("position across the ribbon $y$ [$a$]")
    ax[1].set_ylabel(r"energy $\omega$ [$t$]")

style.save(draw, "ldos_zigzag", npanels=2)
