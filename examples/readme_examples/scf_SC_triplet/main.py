# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
from pyqula import geometry
es = np.linspace(-2.0,2.0,200) # energies of the spectral function
nk = 400 # k-points along the path

# normal state: the ferromagnet without interactions, at the same filling
g = geometry.triangular_lattice() # generate the geometry
h0 = g.get_hamiltonian() # create Hamiltonian of the system
h0.add_exchange([0.,0.,1.]) # add exchange field
h0.set_filling(0.3,nk=40) # put the Fermi energy at zero
(k0,e0,d0) = h0.get_kdos_bands(nk=nk,energies=es,delta=0.03)

# superconducting state
h = g.get_hamiltonian() # create Hamiltonian of the system
h.add_exchange([0.,0.,1.]) # add exchange field
h.setup_nambu_spinor() # initialize the Nambu basis
# perform a superconducting non-collinear mean-field calculation,
# on a k-mesh that resolves the superconducting gap
h = h.get_mean_field_hamiltonian(V1=-1.5,filling=0.3,mf="random",nk=40)
# compute the non-unitarity of the spin-triplet superconducting d-vector
d = h.get_dvector_non_unitarity(nk=40) # non-unitarity of spin-triplet
print("non-unitarity of the d-vector:",d)
print("magnetization:",h.get_magnetization(nk=40))
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
plt.savefig("scf_SC_triplet.png")
plt.show()


