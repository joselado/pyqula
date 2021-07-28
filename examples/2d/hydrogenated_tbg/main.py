# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula importgeometry
from pyqula importhamiltonians
import numpy as np
import klist
import sculpt
import specialgeometry
g = specialgeometry.twisted_bilayer(5)
#g = g.supercell(3)
#g = geometry.honeycomb_lattice()
g = g.remove(i=0)
g.write()
from specialhopping import twisted_matrix
h = g.get_hamiltonian(is_sparse=True,has_spin=False,is_multicell=False,
     mgenerator=twisted_matrix(ti=0.4,lambi=7.0))
import density
from pyqula importscftypes
h.turn_spinful()
h.turn_dense()
mf = scftypes.guess(h,mode="ferro",fun=0.1)
scf = scftypes.selfconsistency(h,nkp=1,filling=0.5,g=1.5,
                mix=0.9,mf=mf,mode="U")
h = scf.hamiltonian
#h.set_filling(nk=3,extrae=1.) # set to half filling + 2 e
#d = density.density(h,window=0.1,e=0.025)
#h.shift_fermi(d)
#h.turn_sparse()
h.get_bands(num_bands=20,operator="sz")







