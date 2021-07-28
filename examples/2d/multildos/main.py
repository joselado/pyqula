# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import ldos
g = geometry.honeycomb_zigzag_ribbon()
h = g.get_hamiltonian(has_spin=True)
ldos.multi_ldos(h,nk=4,es=np.linspace(-1,1,100),delta=0.1,nrep=3)







