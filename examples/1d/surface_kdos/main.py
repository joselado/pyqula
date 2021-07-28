# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
import numpy as np
g = geometry.honeycomb_zigzag_ribbon(50) # create geometry of a zigzag ribbon
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.1)
from pyqula import kdos

edge = np.zeros(h.intra.shape[0]) ; edge += 1.0 ; edge[10:edge.shape[0]] = 0.0
frand = lambda : (-0.5+np.random.random(edge.shape[0]))*edge
kdos.kdos_bands(h,frand=frand)









