# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Conductivity of large disordered systems from a Chebyshev expansion.
# First, the optical conductivity of a gapped honeycomb supercell with onsite
# disorder. The absorption Re sigma_xx is zero below the gap and rises at
# w = 1, and the reactive part Im sigma_xx follows from the absorption at all
# frequencies (Kramers-Kronig): it is negative below the gap, as for a
# dielectric. Second, the DC conductivity of a disordered graphene supercell
# at several Fermi levels, from the Kubo-Greenwood formula at zero
# temperature with one Chebyshev series per Fermi level. Near the Dirac point
# it is about 0.10 in units of e^2/hbar.
import numpy as np
from pyqula import geometry

np.random.seed(1)
g = geometry.honeycomb_lattice().get_supercell(30) # 1800 sites
h = g.get_hamiltonian(has_spin=False,is_sparse=True)
h.add_sublattice_imbalance(0.5) # gap of 1
h.add_onsite(lambda r: 0.6*(np.random.random()-0.5)) # onsite disorder
ws = np.linspace(0.,4.,100) # frequencies
(ws,s) = h.get_optical_conductivity(energies=ws,integration="kpm",
                                    delta=0.05,ntries=4) # full complex tensor

g = geometry.honeycomb_lattice().get_supercell(40) # 3200 sites
h = g.get_hamiltonian(has_spin=False,is_sparse=True)
h.add_onsite(lambda r: 3.0*(np.random.random()-0.5)) # Anderson disorder
fermis = np.linspace(-2.,2.,21) # Fermi levels
sxx = h.get_dc_conductivity(fermi=fermis,delta=0.05,mode="singleshot",
                            component="xx",ntries=4) # sigma_xx(E_F)

import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,2,figsize=(9,4))
ax[0].plot(ws,s[:,0,0].real,label="Re $\\sigma_{xx}$")
ax[0].plot(ws,s[:,0,0].imag,label="Im $\\sigma_{xx}$")
ax[0].set_xlabel("$\\omega$") ; ax[0].legend()
ax[1].plot(fermis,sxx,marker="o")
ax[1].set_xlabel("$E_F$") ; ax[1].set_ylabel("$\\sigma_{xx}$")
plt.tight_layout() ; plt.show()
