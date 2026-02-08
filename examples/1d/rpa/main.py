# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g = geometry.honeycomb_zigzag_ribbon(2)
#g = geometry.lieb_ribbon(1)
g = geometry.bichain()
#g = geometry.chain()
#g = geometry.bisquare_ribbon(2)
h = g.get_hamiltonian()
from pyqula import chi

U = 3.
nk = 50
hmf = h.copy() ; hmf.add_antiferromagnetism(0.5)
#hmf = h.copy() ; hmf.add_exchange([0.,0.,1.])
#h.add_exchange([0.,0.,0.3])
h = h.get_mean_field_hamiltonian(U=U,nk=nk,mf=hmf,filling=0.5)
#exit()
qs = np.linspace(0.,.5,50) # qvectors
energies=np.linspace(.0,3.0,200) # energies
h.get_bands()
chimap = []

import time
t0 = time.time()


for q in qs: # loop over qvectors
    es,chis = h.get_spinchi_ladder(q=q,nk=nk,energies=energies,delta=1e-1)
    cs = [np.trace(c).imag for c in chis]
    cs = np.array(cs)/np.max(cs)
    chimap.append(cs) # store 

t1 = time.time()
print("Time",t1-t0)
#exit()
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,4))
chimap = np.array(chimap) ; chimap = np.abs(chimap) 
chimap = chimap/np.max(chimap)
#cut = .8 ; chimap[chimap>cut] = cut
plt.contourf(qs,energies,chimap.T,levels=100,cmap="Blues_r")
plt.colorbar(label="Im($\\chi_{+-}^{RPA}$)",ticks=[])
plt.xlabel("q-vector") ; plt.ylabel("$\\omega$")
plt.tight_layout()
plt.savefig("spin_chi_rpa.png")


plt.show()

