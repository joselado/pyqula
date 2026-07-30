# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import matplotlib.pyplot as plt
import numpy as np

from pyqula import geometry

# Build a short finite (0d) chain segment to use as the central
# scattering region -- a periodic supercell flagged as dimensionality=0,
# so only the intracell hopping (no wraparound) is kept
g = geometry.chain()
gc = g.get_supercell(5)
gc.dimensionality = 0
hc = gc.get_hamiltonian() # finite, 5-site central region

# Two leads: a plain normal chain, and a chain with s-wave pairing
h_normal = g.get_hamiltonian()
h_sc = g.get_hamiltonian()
delta = 0.05
h_sc.add_swave(delta)

# attach the normal lead to site 0 and the superconducting lead to the
# last site of the central region -- returns a Heterostructure, so all
# its usual methods (didv, landauer, get_dos...) work unmodified
nsites = len(gc.r)
ht = hc.get_central_heterostructure(0, nsites-1, left=h_normal, right=h_sc)
ht.delta = 1e-4

es = np.linspace(-3*delta, 3*delta, 100)
Gs = [ht.didv(energy=e) for e in es] # Andreev conductance (auto-dispatches to the BdG/smatrix formula since ht.has_eh is True)

plt.plot(es/delta, Gs)
plt.xlabel("Energy [$\\Delta$]")
plt.ylabel("dIdV [$e^2/h$]")
plt.show()
