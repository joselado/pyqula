# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import geometry

g = geometry.triangular_lattice()
g = g.get_supercell([np.sqrt(3),np.sqrt(3)]) # supercell
g.write(nrep=3)
