# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Entanglement entropy of a critical one-dimensional chain
# The Hamiltonian is cut into a ring of L unit cells, and the entropy of
# an arc of l cells is computed from the one-particle correlation matrix.
# For a gapless free-fermion chain conformal field theory predicts
#   S(l) = (c/3) ln[(L/pi) sin(pi l/L)] + const
# with central charge c=1, which is what the fit below recovers.

import numpy as np
from pyqula import geometry

g = geometry.chain() # one-dimensional chain
h = g.get_hamiltonian(has_spin=False) # spinless, half filled

L = 62 # number of unit cells of the ring
# rings with L divisible by 4 have levels exactly at the Fermi energy,
# which makes the ground state degenerate (the entropy then raises)

ls = range(4,L//2+1) # lengths of the region
ss = [h.get_entanglement_entropy(nsuper=L, # cells in the ring
                                 region=list(range(l)) # cells in region A
                                 ) for l in ls]

chord = [np.log(L/np.pi*np.sin(np.pi*l/L)) for l in ls] # CFT scaling variable
(slope,offset) = np.polyfit(chord,ss,1) # linear fit
print("Central charge c =",round(3*slope,4)) # should be 1

# entanglement spectrum of half of the ring
xi = h.get_entanglement_spectrum(nsuper=L,region=0.5)

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(chord,ss,marker="o",linestyle="none",label="pyqula")
plt.plot(chord,np.array(chord)*slope+offset,c="black",
         label="c = "+str(round(3*slope,3)))
plt.xlabel("ln[(L/$\\pi$) sin($\\pi l/L$)]") ; plt.ylabel("Entanglement entropy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(len(xi)),xi,marker="o",linestyle="none")
plt.axhline(0.,c="black",linewidth=0.5)
plt.xlabel("Level index") ; plt.ylabel("Entanglement spectrum $\\xi_n$")
plt.ylim([-10,10])

plt.tight_layout()
plt.show()
