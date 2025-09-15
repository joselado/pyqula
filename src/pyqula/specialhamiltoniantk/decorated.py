import numpy as np
from .. import geometry
from .. import potentials

def decorated_triangular(t1=1.0,t2=1.0,mode="th",**kwargs):
    """Return the Hamiltonian of a decorated honeycomb lattice,
    namely a triangular lattice with sqrt(3) dimerization"""
    g = geometry.triangular_lattice_tripartite()
    print(g.a1,g.a2)
    if mode=="th":
        g.shift(-g.r[2])
        pot = potentials.commensurate_potential(g,n=6,minmax=[-.2,1.])
    elif mode=="tri":
        g.center()
        pot = potentials.commensurate_potential(g,n=3,minmax=[-.2,1.])
    g.get_fractional()
    h = g.get_hamiltonian(has_spin=False,**kwargs)
    pot = pot.redefine(lambda v: 1.*(v>0))
#    pot = 1.-pot/2.
    pot = t1*pot + (1.-pot)*t2
    h.add_strain(pot,mode="scalar")
    h.add_onsite(pot)
    return h


