# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
from pyqula import parallel
#parallel.numba_cores = 4


g = geometry.bichain()
h = g.get_hamiltonian()

U = 3.  
nk = 20
hmf = h.copy() ; hmf.add_exchange(lambda r: np.random.random(3)-.5) #; hmf.add_antiferromagnetism(0.5)
#hmf = h.copy() ; hmf.add_exchange([0.,0.,1.])
#h.add_exchange([0.,0.,0.3])
h = h.get_mean_field_hamiltonian(U=U,nk=nk,mf=hmf,filling=0.5)
qs = np.linspace(0.,.5,20) # qvectors
energies=np.linspace(.0,5.,100) # energies
h.get_bands(operator="sz")
print("Mx",h.get_vev("sx"))
print("My",h.get_vev("sy"))
print("Mz",h.get_vev("sz"))

import time
t0 = time.time()
(qs,es,chis) = h.get_qdos_iets(energies = np.linspace(0.,5.0,400),
                               nq=20,nk=nk,delta=1e-2)
t1 = time.time()
qs = np.unique(qs,axis=0)
es = np.unique(es)
chimap = chis.reshape((len(qs),len(es))).T


print("Time",t1-t0)
#exit()
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,4))
vmax = np.percentile(chimap,98)
chimap[chimap>vmax] = vmax
plt.contourf(range(len(qs)),
             es,np.sqrt(np.abs(chimap)),levels=100,cmap="Blues_r")
plt.colorbar(label="Im($\\chi_{+-}^{RPA}$)",ticks=[])
plt.xlabel("q-vector [$\\pi$]") ; plt.ylabel("$\\omega$")
plt.tight_layout()
plt.savefig("spin_chi_rpa.png")


plt.show()



