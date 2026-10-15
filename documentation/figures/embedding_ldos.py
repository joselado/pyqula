import numpy as np
import style
from pyqula import geometry
from pyqula import embedding
g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # pristine, infinite Hamiltonian
hv = h.copy()
hv.add_onsite(lambda r: 1.0 if r[0]<0.01 else 0.0) # a single-site onsite defect
eb = embedding.Embedding(h,m=hv) # embed the defect in the infinite system
(x,y,d) = eb.get_ldos(energy=0.0,delta=1e-2,nsuper=200,nk=400) # LDOS around the defect

# the same LDOS at many energies, on 40 sites and with a finer k-mesh
es = np.linspace(-2.6,2.6,105)
ldos = np.array([eb.get_ldos(energy=e,delta=1e-2,nsuper=40,nk=2000)[2] for e in es])
x40 = eb.get_ldos(energy=0.,delta=1e-2,nsuper=40,nk=2000)[0] # positions of the 40 sites

def draw(ax, c):
    m = ax.pcolormesh(x40, es, ldos, cmap=c.cmap, vmin=0, vmax=1, shading="nearest")
    style.colorbar(m, ax, "local density of states")
    ax.set_xlabel("position $x$")
    ax.set_ylabel("energy [$t$]")

style.save(draw, "embedding_ldos")
