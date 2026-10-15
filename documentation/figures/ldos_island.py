import numpy as np
import style
from pyqula import islands

g = islands.get_geometry(name="honeycomb",n=3,nedges=3) # get an island
h = g.get_hamiltonian() # get the Hamiltonian
(x,y,es,ldos) = h.get_multildos(projection="atomic") # get the LDOS

def draw(ax, c):
    for a, e0 in zip(ax, [0.0, 0.67]): # zero modes, and the first level above
        i = np.argmin(np.abs(es-e0)) # the frame closest to e0
        im = a.tripcolor(x, y, ldos[i], cmap=c.cmap, shading="gouraud")
        style.colorbar(im, a, "LDOS per unit area")
        a.scatter(g.x, g.y, s=3, color=c.ink2)
        a.set_aspect("equal")
        a.set_title(r"$\omega=%.2f\,t$" % es[i])
        a.set_xlabel("$x$ [$a$]")
        a.set_ylabel("$y$ [$a$]")

style.save(draw, "ldos_island", npanels=2)
