# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np

# Build an antiferromagnetic Hubbard mean-field chain, then locate the
# poles of its full spin RPA kernel (1 - U*chi) along the Brillouin zone:
# these poles are the magnon bands, i.e. the collective spin-wave modes of
# the ordered state.

g = geometry.bichain()
h = g.get_hamiltonian()
hmf = h.copy() ; hmf.add_antiferromagnetism(0.5) # symmetry-breaking seed
U = 3.
h = h.get_mean_field_hamiltonian(U=U,nk=100,mf=hmf,filling=0.5)
print("Mz sublattice",h.get_vev("sz"))

nq = 40 # number of q-points along the path
energies = np.linspace(0.01,4.,200) # frequency window (omega=0 excluded,
                                     # it is a trivial root of every kernel)
qs,ws,gammas = h.get_magnon_bands(nq=nq,energies=energies,delta=2e-2,nk=100)

# each returned pole has a signed residual imaginary part (gammas): the
# smaller its magnitude, the sharper/better-defined the collective mode.
# Keep only the well-defined ones for a clean dispersion plot.
sharp = np.abs(gammas) < 0.05
qs,ws,gammas = qs[sharp],ws[sharp],gammas[sharp]

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,4))
plt.scatter(qs/(nq-1),ws,c=gammas,cmap="inferno_r",s=15)
plt.colorbar(label="$\\gamma$ (mode broadening)")
plt.xlabel("q-vector [$\\pi$]") ; plt.ylabel("$\\omega$ (magnon energy)")
plt.tight_layout()
plt.savefig("magnon_bands.png")
plt.show()
