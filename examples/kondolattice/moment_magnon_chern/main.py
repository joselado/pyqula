# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Chern number of the magnon of the local moments of a Kondo lattice.
#
# The magnon of the moments is a zero of the RPA kernel
#   K(q,w) = 1/chi(q,w) - U'(q) ,
# a matrix between the moments of the unit cell, with chi the mean-field
# susceptibility of the moments and U' the residual interaction of
# Chen, Stoudenmire, Komijani and Coleman, arXiv:2302.09701. At the energy
# of the magnon K has a null vector, the amplitude of the magnon on the
# moments, and its Berry fluxes on a mesh of q give the Chern number.
#
# One moment per cell gives zero by construction, so we take the honeycomb
# lattice, two moments per cell, with a Haldane term t2 on the conduction
# electrons, J=4 at half filling and U'=-4. One magnon is bound below the
# continuum, and its Chern number is the sign of t2: +1 for t2=0.2 and -1
# for t2=-0.2. This is measured, not derived, and U' is a parameter of the
# model. The window (0.3,1.43) holds the band, from 0.590 to 1.357, below
# the next band, which starts at 1.506, and the continuum at 1.695.

import numpy as np
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

def kondo(t2):
    """Mean field of the honeycomb Kondo lattice with a Haldane term t2"""
    hc = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    hc.add_haldane(t2) # breaks time reversal of the conduction electrons
    h = KondoLatticeHamiltonian(hc) # two moments per unit cell
    return h.get_mean_field_hamiltonian(J=4.0, filling=0.5, nk=12,
            mf=([1.0+0j]*2, [0.0]*2))

for t2 in [0.2,-0.2]:
    h2 = kondo(t2)
    r = h2.get_moment_magnon_chern((0.3, 1.43), U=-4.0, nk=8, nq=12,
            full_output=True)
    print("t2", t2, "| C =", np.round(r.chern, 8), "| band from",
          np.round(r.energies.min(), 3), "to", np.round(r.energies.max(), 3),
          "| bound", np.round(r.bound, 3))

# The magnon is a spin triplet. The raising channel, the default, holds
# one copy of it and gives 1 on every mesh below. The full channel holds
# three copies with the same vector, so the flux of each plaquette is three
# times larger, and the integer it returns depends on the mesh: 3, 1, 1 and
# 3 at nq = 6, 8, 10 and 12 for t2=0.2, with no error.
h2 = kondo(0.2)
for nq in [6, 8, 10, 12]:
    cr = h2.get_moment_magnon_chern((0.3, 1.43), U=-4.0, nk=8, nq=nq)
    cf = h2.get_moment_magnon_chern((0.3, 1.43), U=-4.0, nk=8, nq=nq,
            channel="full")
    print("nq", nq, "| raising channel C =", np.round(cr, 6),
          "| full channel C =", np.round(cf, 6))
