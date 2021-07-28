# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import dos
g = geometry.diamond_lattice()
h = g.get_hamiltonian()
h.turn_dense()
h.add_antiferromagnetism(1.)
#dos.dos(h,nk=1000,delta=0.01,random=True,energies=np.linspace(-6.0,6.0,100))
dos.autodos(h,nk=100,auto=True,delta=0.1,energies=np.linspace(-6.0,6.0,1000))
h.get_bands()







