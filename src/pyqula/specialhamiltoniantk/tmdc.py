import numpy as np
from .. import geometry

def doped_MoS2():
    """Hamiltonian of doped MoS2"""
    g = geometry.honeycomb_lattice() # geometry
    h = g.get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.6)
    h.add_soc(0.05)
    h.set_filling(0.501,nk=20)
    return h


