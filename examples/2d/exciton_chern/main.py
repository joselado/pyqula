# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Chern number of a Frenkel exciton band, from the Bethe-Salpeter equation.
#
# An exciton at momentum Q is a sum of electron-hole pairs, an electron at
# k+Q and a hole at k. Written in the orbitals,
#   Phi_Q(k;a,b) = sum_vc A_vc(k) u_c,k+Q(a) conj(u_v,k(b)) ,
# it does not depend on the phases of the Bloch states, and the Chern
# number of an exciton band is the sum of its Berry fluxes through the
# plaquettes of a mesh of Q, over 2 pi, with the sign of h.get_chern().
#
# The model is the kagome lattice with the hoppings inside the up
# triangles only, so each triangle is a molecule with three levels, one of
# them filled, and a flux phi through it. A density interaction V1=1
# between neighbors binds the electron and the hole on one molecule and
# moves the bound pair to the next molecules, a Frenkel exciton. The
# lowest exciton band has C=-1 for phi=0.3, and the opposite flux gives
# C=+1. Without flux the molecule has time-reversal symmetry, the two
# lowest excitons touch at Q=0, and the calculation raises an error.

import numpy as np
from pyqula import geometry
from pyqula.multihopping import MultiHopping
from pyqula.bsetk.interaction import density_interaction

def frenkel(phi):
    """Kagome lattice of decoupled up triangles with a flux phi"""
    H = np.zeros((3,3),dtype=np.complex128) # one up triangle, a molecule
    for (i,j) in [(0,1),(1,2),(2,0)]:
        H[j,i] = -np.exp(1j*phi/3.) # hopping with a Peierls phase
        H[i,j] = np.conj(H[j,i])
    H += 0.5*np.identity(3) # one level below the Fermi energy, two above
    g = geometry.kagome_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.set_multihopping(MultiHopping({(0,0,0):H})) # hopping in the cell only
    return h

for phi in [0.3,-0.3,0.0]:
    h = frenkel(phi)
    W = density_interaction(h,V1=1.0) # density interaction between neighbors
    try:
        r = h.get_exciton_chern(bands=[0],nq=8,V=W,nk=6,tda=True,
                                full_output=True) # lowest exciton band
        print("phi",phi,"| C =",np.round(r.chern,8),
              "| gap",np.round(r.gap,4),"| bound",np.round(r.bound,4))
    except ValueError as error: # the two lowest excitons touch
        print("phi",phi,"| no Chern number:",str(error)[0:60],"...")
