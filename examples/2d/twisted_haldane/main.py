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
g.write()
from specialhopping import twisted_matrix
h = g.get_hamiltonian(is_sparse=True,has_spin=False,is_multicell=False,
     mgenerator=twisted_matrix(ti=0.4,lambi=7.0))
#h.turn_spinful()
h.add_haldane(0.05)
h.add_sublattice_imbalance(0.5)
#h.add_kane_mele(0.1)
#h.add_rashba(lambda r:  0.1*np.sign(r[2]))
h.shift_fermi(-0.3)
#h.add_sublattice_imbalance(0.1)
#h.shift_fermi(lambda r: r[2]*0.1)
import density
#h.set_filling(nk=3,extrae=1.) # set to half filling + 2 e
#d = density.density(h,window=0.1,e=0.025)
#h.shift_fermi(-0.4)
#h.turn_sparse()
#h.get_bands(num_bands=20)
#exit()
import topology
#print(h.get_gap())
#exit()
h.turn_dense()
#topology.write_berry(h)
import parallel
parallel.cores = 6
topology.mesh_chern(h,nk=10)







