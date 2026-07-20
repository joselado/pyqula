# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import geometry
g = geometry.honeycomb_lattice()

# Default: density matrix computed by exact diagonalization on a k-mesh
h1 = g.get_hamiltonian()
hscf1,e1 = h1.get_mean_field_hamiltonian(U=2.0,filling=0.5,mf="antiferro",
        nk=8,maxerror=1e-4,return_total_energy=True)
print("integration='ed'   total energy = ",e1)

# Alternative: each required density-matrix entry is integrated over the
# BZ with qutecipy (tensor cross interpolation) instead of averaged over
# the k-mesh -- same SCF loop, just a different density-matrix backend
h2 = g.get_hamiltonian()
hscf2,e2 = h2.get_mean_field_hamiltonian(U=2.0,filling=0.5,mf="antiferro",
        nk=8,maxerror=1e-4,return_total_energy=True,integration="qtci")
print("integration='qtci' total energy = ",e2)

(k1,e1b) = hscf1.get_bands()
(k2,e2b) = hscf2.get_bands()

import matplotlib.pyplot as plt

plt.scatter(k1,e1b,label="ed")
plt.scatter(k2,e2b,label="qtci")
plt.legend()
plt.xlabel("k-path") ; plt.ylabel("Energy")
plt.show()
