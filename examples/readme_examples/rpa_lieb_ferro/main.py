# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry ; import numpy as np
g = geometry.lieb_ribbon(2) # Lieb lattice ribbon
h = g.get_hamiltonian() # generate Hamiltonian
h = h.get_mean_field_hamiltonian(U=3.,mf="antiferro",filling=0.5) # perform SCF
energies=np.linspace(.0,1.,400) # energies
# RPA many-body spin spectral function
(qs,es,chis) = h.get_qdos_iets(energies = energies,nq=100,nk=10,
                               delta=1e-2,qpath=["G","X"])



qs = np.unique(qs,axis=0)
es = np.unique(es)
chimap = chis.reshape((len(qs),len(es))).T
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,4))
vmax = np.percentile(chimap,99)
chimap[chimap>vmax] = vmax
plt.contourf(range(len(qs)),
             es,np.sqrt(np.abs(chimap)),levels=100,cmap="Blues_r")
plt.colorbar(label="Im($\\chi_{+-}^{RPA}$)",ticks=[])
plt.xlabel("q-vector") ; plt.ylabel("$\\omega$")
plt.xticks([])
plt.tight_layout()
plt.savefig("spin_chi_rpa.png")


plt.show()




