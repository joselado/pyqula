# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Anomalous Hall conductivity of the Haldane model with a staggered onsite
# energy, against the Fermi energy. In the gap it is the Chern number, C=1.
# Inside the bands it is the Berry curvature integrated over the Fermi sea,
# and it is not quantized.

from pyqula import geometry
import numpy as np
g = geometry.honeycomb_lattice() # honeycomb lattice
h = g.get_hamiltonian(has_spin=False) # spinless Hamiltonian
h.add_haldane(0.1) # Haldane coupling, it opens a topological gap
h.add_sublattice_imbalance(0.2) # staggered onsite, the gap is now -0.25 to 0.25
fermis = np.linspace(-3.2,3.2,161) # Fermi energies, across the whole band
# the triangle integration is the default, so a coarse mesh is enough in a metal
s = h.get_anomalous_hall_conductivity(fermi=fermis,nk=100,T=1e-3)
np.savetxt("HALL.OUT",np.array([fermis,s]).T)

import matplotlib.pyplot as plt
plt.plot(fermis,s)
plt.xlabel("Fermi energy [t]")
plt.ylabel("$\\sigma_{xy}$ [$e^2/h$]")
plt.show()
