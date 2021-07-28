# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import specialgeometry
from pyqula.specialhopping import twisted_matrix
from pyqula import meanfield
from pyqula import groundstate
from pyqula import geometry


g = specialgeometry.twisted_bilayer(4)
g = geometry.honeycomb_lattice() ; g  = g.supercell(3)
h = g.get_hamiltonian(is_sparse=True,has_spin=False,is_multicell=False,
     mgenerator=twisted_matrix(ti=0.0,lambi=7.0))
mf = meanfield.guess(h,"dimerization")
scf = meanfield.Vinteraction(h,nk=1,filling=0.5,V1=2.0,V2=1.0, mix=0.3,mf=mf)
h = scf.hamiltonian # get the Hamiltonian
groundstate.hopping(h,nrep=1,skip = lambda r1,r2: r1[2]*r2[2]<0) # write three replicas
h.get_bands() # calculate band structure
h.write_onsite()







