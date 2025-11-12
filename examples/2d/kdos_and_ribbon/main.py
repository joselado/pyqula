# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import matplotlib.pyplot as plt
import numpy as np
from pyqula import geometry
from pyqula import kdos
g = geometry.triangular_lattice()
h = g.get_hamiltonian()
h.add_onsite(2.)
h.add_rashba(0.5)
h.add_exchange([0.,0.,0.5])
h.add_swave(0.1)

# energy window and momenta to show
ng = 100 # number of energies
es0 = np.linspace(-0.2,0.2,ng)
klist = [(k,0,0) for k in np.linspace(-0.5,0.5,ng)]
####


from pyqula import ribbon

# compute the ribbon
hr = ribbon.bulk2ribbon(h,n=100)
(ksr,esr,csr) = hr.get_bands(operator="yposition",num_bands=40,kpath=klist)

# compute semiinfinite system
(ks,es,db,ds) = h.get_kdos(energies=es0,kpath=klist,delta=1./ng)
# normalize 
db = db/np.max(db)
ds = ds/np.max(ds)


# now plot both

fig = plt.figure(figsize=(8,4))

plt.subplot(1,2,1) ; plt.title("Ribbon (includes 2 edge)")
plt.scatter(ksr,esr,c=csr,cmap="rainbow")
plt.ylim([min(es0),max(es0)]) ; plt.ylabel("Energy")
plt.xticks([]) ; plt.xlabel("Momentum")
plt.colorbar(ticks=[],label="Uppe/bulk/lower edge")


plt.subplot(1,2,2) ; plt.title("Semiinfinite (just one edge)")
ks = np.unique(ks) ; es = np.unique(es) ; db = db.reshape((len(es),len(ks))).T
plt.contourf(ks,es,db,vmax=0.6,cmap="inferno",levels=60)
plt.xlabel("k") ; plt.ylabel("E")
plt.ylim([min(es),max(es)]) ; plt.ylabel("Energy")
plt.xticks([]) ; plt.xlabel("Momentum")
plt.colorbar(ticks=[],label="DOS")

plt.tight_layout()

plt.show()


