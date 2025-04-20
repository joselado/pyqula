import numpy as np
from .. import geometry
from .. import potentials

def decorated_triangular(t1=1.0,t2=1.0):
    """Return the Hamiltonian of a decorated honeycomb lattice,
    namely a triangular lattice with sqrt(3) dimerization"""
    g = geometry.triangular_lattice_tripartite()
    g.shift(-g.r[2])
    h = g.get_hamiltonian(has_spin=False)
    pot = potentials.commensurate_potential(g,n=6,minmax=[-.2,1.])
    pot = pot.redefine(lambda v: 1.*(v>0))
#    pot = 1.-pot/2.
    pot = t1*pot + (1.-pot)*t2
    h.add_strain(pot,mode="scalar")
    h.add_onsite(pot)
    return h


