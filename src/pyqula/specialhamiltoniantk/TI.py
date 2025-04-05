
from .. import geometry
from .. import films
import numpy as np


def QSH3D_film(W=20,dt=0.6,soc = 0.1,g=None):
    """Return the Hamiltonian of a 3D quantum spin Hall insulator"""
    if g is None: # no geometry provided
        g0 = geometry.diamond_lattice_minimal()
        g = films.geometry_film(g0,nz=W)
    def tij(ri,rj):
        dr = ri-rj
        dr2 = dr.dot(dr)
        if 0.99<dr2<1.01:
            if 0.99<np.abs(dr[2])<1.01:
                return 1.0 + dt
            else: return 1.0
        else: return 0.
    h = g.get_hamiltonian(tij=tij,is_sparse=True)
#    from ..specialhopping import entry2matrix
#    mgen = entry2matrix(tij) # create an mgenerator
#    h = g.get_hamiltonian(mgenerator=mgen)
    h.add_kane_mele(soc)
    return h
