# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g = geometry.honeycomb_zigzag_ribbon(1)
#g = geometry.lieb_ribbon(1)
#g = geometry.bichain()
#g = geometry.chain()
h = g.get_hamiltonian()
from pyqula import chi

U = 1.
nk = 100
hmf = h.copy() ; hmf.add_antiferromagnetism(0.5)
#hmf = h.copy() ; hmf.add_exchange([0.,0.,1.])
#h.add_exchange([0.,0.,0.3])
h = h.get_mean_field_hamiltonian(U=U,nk=nk,mf=hmf,filling=0.5)

qs = np.linspace(0.,.5,50) # qvectors
energies=np.linspace(.0,6.0,200) # energies
h.get_bands()
chimap = []

import time
t0 = time.time()

for q in qs: # loop over qvectors
    es,chis = chi.spinchi_ladder(h,q=q,nk=nk,energies=energies,delta=2e-2)
    cs = [np.trace(c).imag for c in chis]
    cs = np.array(cs)/np.max(cs)
    chimap.append(cs) # store 

t1 = time.time()
print("Time",t1-t0)
#exit()
import matplotlib.pyplot as plt
chimap = np.array(chimap) ; chimap = np.abs(chimap) 
chimap = chimap/np.max(chimap)
#cut = .8 ; chimap[chimap>cut] = cut
plt.contourf(qs,energies,chimap.T,levels=100,cmap="rainbow")
plt.colorbar()


plt.show()

