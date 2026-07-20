# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import hamiltonians
from pyqula import klist
from pyqula import sculpt
from pyqula import specialgeometry
g = specialgeometry.twisted_bilayer(8)
#g = geometry.honeycomb_lattice()
g.write()
from pyqula import specialhopping
h = g.get_hamiltonian(is_sparse=True,has_spin=False,is_multicell=False,
     mgenerator=specialhopping.twisted_matrix(ti=0.5,lambi=7.0))
h.turn_dense()
#def ff(r):
#    return 0.2*r[2]
    
#h.shift_fermi(ff)
h.turn_spinful()
h.turn_dense()
h.add_sublattice_imbalance(0.5)
h.add_kane_mele(0.03)
#h.get_bands(num_bands=40)
#exit()
from pyqula import meanfield
mf = meanfield.guess(h,"antiferro",0.1)
U = 2.0
filling = 0.5 + 1./h.intra.shape[0] # plus two electrons
nk = 1
scf = meanfield.hubbardscf(h,nk=nk,filling=filling,U=U,
                mix=0.9,mf=mf,maxite=1)
(k,e,c) = scf.hamiltonian.get_bands(num_bands=40,operator="sz")

import matplotlib.pyplot as plt

plt.scatter(k,e,c=c,cmap="bwr")
plt.colorbar(label="Sz")
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.show()
#print(scf.total_energy-scf1.total_energy)







