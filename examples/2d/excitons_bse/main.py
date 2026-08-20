# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Excitons from the Bethe-Salpeter equation, on top of a mean field.
#
# An independent-particle band structure can only absorb light above its
# gap. Switching on the electron-hole interaction binds the pair, and the
# exciton appears below the gap by its binding energy. Here we take a
# gapped honeycomb semiconductor and watch the lowest exciton detach from
# the absorption edge as the Coulomb tail is turned up.

import numpy as np
import matplotlib.pyplot as plt
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_sublattice_imbalance(1.0) # open a gap, a two-band semiconductor

def coulomb(e2):
    """A Coulomb tail with a soft cutoff at short distance"""
    return lambda r1,r2: e2/np.sqrt((r1-r2).dot(r1-r2)+0.25)

e2s = np.linspace(0.,1.2,7) # interaction strengths to scan
gap,lowest,binding = None,[],[]
for e2 in e2s:
    W = density_interaction(h,Vr=coulomb(e2)) # the electron-hole interaction
    bse = h.get_bse(V=W,nk=8) # solve the full (non-Tamm-Dancoff) BSE
    gap = np.min(bse.pairs.dE) # lowest independent-particle transition
    lowest.append(bse.get_energies()[0].real)
    binding.append(bse.get_binding_energies()[0].real)
    print("e2 = %4.2f   lowest exciton = %.4f   binding = %.4f"
            %(e2,lowest[-1],binding[-1]))

plt.plot(e2s,lowest,marker="o",label="lowest exciton")
plt.axhline(gap,color="k",ls="--",label="single-particle gap")
plt.xlabel("Interaction strength") ; plt.ylabel("Energy")
plt.legend() ; plt.tight_layout()
plt.show()
