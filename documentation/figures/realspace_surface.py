import numpy as np
import style
from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
(k,e,ds,db) = kdos.surface(h) # surface spectral function
k,e = k.reshape(100,100),e.reshape(100,100) # 100 momenta times 100 energies
ds,db = ds.reshape(100,100),db.reshape(100,100) # surface and bulk weights

def draw(ax, c):
    for (axi,d,name) in zip(ax,[ds,db],["surface","bulk"]):
        im = axi.pcolormesh(k, e, d, cmap=c.cmap, vmin=0, vmax=np.percentile(ds,99))
        style.colorbar(im, axi, f"{name} spectral weight")
        axi.set_xlabel(r"momentum along the edge [$2\pi/a$]")
    ax[0].set_ylabel("energy [$t$]")

style.save(draw, "realspace_surface", npanels=2)
