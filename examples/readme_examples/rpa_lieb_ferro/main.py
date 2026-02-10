# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry ; import numpy as np
g = geometry.lieb_ribbon(2) # Lieb lattice ribbon
h = g.get_hamiltonian() # generate Hamiltonian
h = h.get_mean_field_hamiltonian(U=3.,mf="antiferro",filling=0.5) # perform SCF
qs = np.linspace(0.,.5,50) # qvectors
energies=np.linspace(.0,1.,400) # energies

chimap = [] # storage for the results
for q in qs: # loop over qvectors
    es,chis = h.get_spinchi_ladder(q=q,energies=energies,delta=2e-2) 
    # compute RPA tensor
    cs = [np.trace(c).imag for c in chis] # imaginary part of the trace
    chimap.append(cs) # store 

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,4))
chimap = np.array(chimap) ; chimap = np.abs(chimap) 
chimap = chimap/np.max(chimap)
#cut = .8 ; chimap[chimap>cut] = cut
plt.contourf(qs,energies,chimap.T,levels=100,cmap="Blues_r")
plt.colorbar(label="Im($\\chi_{+-}^{RPA}$)",ticks=[])
plt.xlabel("q-vector [$\\pi$]") ; plt.ylabel("$\\omega$")
plt.tight_layout()
plt.savefig("spin_chi_rpa.png")


plt.show()

