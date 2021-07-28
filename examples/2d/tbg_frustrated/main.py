# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import hamiltonians
import numpy as np
from pyqula import specialgeometry
g = specialgeometry.twisted_bilayer(1)
#g = geometry.honeycomb_lattice()
g.write()
#g = geometry.read()
#exit()
from pyqula.specialhopping import twisted,twisted_matrix
h = g.get_hamiltonian(is_sparse=True,has_spin=True,is_multicell=False,
     mgenerator=twisted_matrix(ti=0.4,lambi=7.0))
h.turn_dense()
g.shift(g.a2/4.+g.a1/8.) #; g.center()
g.write()
#exit()
h.add_zeeman([0.0,0.,0.]) # Zeeman field
h.generate_spin_spiral(qspiral=[1./3.,1./3.,0.])
#d = density.density(h,window=0.1,e=0.025)
#h.shift_fermi(d)
h.turn_sparse()
h.get_bands(num_bands=20)







