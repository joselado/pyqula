# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import hamiltonians
import numpy as np
from pyqula import klist
from pyqula import sculpt
from pyqula import specialgeometry
from pyqula import parallel
parallel.cores = 7

g = specialgeometry.multilayer_graphene(l=[0,1])
g = specialgeometry.parse_twisted_multimultilayer([["AB","BA"],[0,1]],n=3)
g.write()
exit()
#g = g.supercell(3)
#g = geometry.honeycomb_lattice()
from pyqula.specialhopping import twisted_matrix
h = g.get_hamiltonian(is_sparse=True,has_spin=False,is_multicell=False,
     mgenerator=twisted_matrix(ti=0.4,lambi=7.0))
#h.set_filling(nk=3,extrae=1.) # set to half filling + 2 e
#d = density.density(h,window=0.1,e=0.025)
#h.shift_fermi(d)
#h.turn_sparse()
from pyqula import algebra
algebra.accelerate = False
h.get_bands(nk=100)







