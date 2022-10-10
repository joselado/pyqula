# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g = geometry.honeycomb_zigzag_ribbon(4) # create geometry of a zigzag ribbon
# remove upper row of atoms
ze = np.max(g.r[:,1]) ; g = g.remove(lambda r: abs(r[1]-ze)<1e-3)
# remove lower row of atoms
ze = np.min(g.r[:,1]) ; g = g.remove(lambda r: abs(r[1]-ze)<1e-3)
g.write(nrep=10) # write geometry
h = g.get_hamiltonian() # create hamiltonian of the system
h.get_bands() # get bands







