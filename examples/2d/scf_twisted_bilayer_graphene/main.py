# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import hamiltonians
from pyqula import klist
from pyqula import sculpt
from pyqula import specialgeometry
from pyqula import scftypes
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
g = 2.0
filling = 0.5 + 1./h.intra.shape[0] # plus two electrons
nk = 1
scf = scftypes.hubbardscf(h,nkp=nk,filling=filling,g=g,
                mix=0.9,mf=mf,maxite=1)
#scf1 = scftypes.selfconsistency(h,nkp=nk,filling=0.5,g=g,
#                mix=0.9,mf=mf)
scf.hamiltonian.get_bands(num_bands=40,operator="sz")
#print(scf.total_energy-scf1.total_energy)







