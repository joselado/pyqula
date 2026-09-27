# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
es = np.linspace(-2.0,2.0,200) # energies of the spectral function
nk = 400 # k-points along the path

# normal state: the same lattice without interactions, at the same filling
g = geometry.triangular_lattice() # geometry of a triangular lattice
h0 = g.get_hamiltonian() # get the Hamiltonian
h0.set_filling(0.45,nk=40) # put the Fermi energy at zero
(k0,e0,d0) = h0.get_kdos_bands(nk=nk,energies=es,delta=0.03)

# superconducting state
h = g.get_hamiltonian()  # get the Hamiltonian
h.setup_nambu_spinor() # setup the Nambu form of the Hamiltonian
# perform SCF, on a k-mesh that resolves the superconducting gap
h = h.get_mean_field_hamiltonian(U=-2.0,filling=0.45,mf="swave",nk=40)
print("Delta =",np.abs(h.extract("swave")[0])) # s-wave order parameter
print("filling =",h.get_filling(nk=40)) # filling of the superconductor
# electron spectral-function
(k,e,d) = h.get_kdos_bands(operator="electron",nk=nk,energies=es,delta=0.03)

import matplotlib.pyplot as plt
fig,axs = plt.subplots(2,1,figsize=(8,5))
for (ax,dk) in zip(axs,[d,d0]): # superconducting (top) and normal (bottom)
    m = dk.reshape(nk,len(es)).T # (energy,k) grid
    ax.imshow(m,origin="lower",aspect="auto",cmap="inferno",
              extent=[0.,1.,es[0],es[-1]],vmax=np.percentile(m,99))
    ax.set_ylabel("E",fontsize=20) ; ax.set_xticks([])
plt.tight_layout()
plt.savefig("scf_SC.png")
plt.show()


