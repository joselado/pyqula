import style
from pyqula import geometry
import numpy as np
g = geometry.honeycomb_lattice().get_supercell(80) # 12800 sites
g.dimensionality = 0 # a finite island with zigzag edges
h = g.get_hamiltonian(is_sparse=True,has_spin=False) # keep it sparse
(x,y,es,ldos) = h.get_multildos(energies=np.linspace(-1.,1.,41), # energies
                    mode="KPM", # Chebyshev expansion of every site
                    delta=0.1, # energy resolution, half width of the peak of a level
                    write=False) # return the maps without writing them

tip = x < x.min() + 35 # zoom on the left corner of the island

def draw(axs, c):
    for (ax,i) in zip(axs, [20, 40]): # E = 0 and E = 1
        m = ax.scatter(x[tip], y[tip], c=ldos[i][tip], s=5, cmap=c.cmap, vmin=0)
        style.colorbar(m, ax, "local density of states")
        ax.set_title("$E=%.0f\\,t$" % es[i])
        ax.set_aspect("equal")
        ax.set_axis_off()

style.save(draw, "kpm_ldos", npanels=2)
