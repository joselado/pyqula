import numpy as np
import style
from pyqula import geometry

g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_soc(0.05) # Kane-Mele spin-orbit coupling opens a topological gap
h.add_rashba(0.1) # break z-mirror symmetry to expose the edge states
(k,e,ds,db) = h.get_surface_kdos(delta=1e-2) # surface and bulk spectral functions

nk, ne = len(np.unique(k)), len(np.unique(e))
(k,e,ds,db) = [a.reshape(nk,ne) for a in (k,e,ds,db)] # k is the outer index

def draw(ax, c):
    for a, d, name in zip(ax, [ds, db], ["surface $A_s$", "bulk $A_b$"]):
        im = a.pcolormesh(k, e, d, cmap=c.cmap, shading="auto", vmax=np.percentile(ds,98))
        style.colorbar(im, a, name+r"$(k,\omega)$")
        a.set_xlabel(r"momentum along the edge $k$ [$2\pi/L$]")
        a.set_ylabel(r"energy $\omega$ [$t$]")

style.save(draw, "spectral_surface", npanels=2)
