# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(.05) # add Haldane hopping to the Hamiltonian
(k,e) = h.get_bands() # get the bandstructure (also written to BANDS.OUT)
(kb,b) = h.get_berry_curvature() # get the Berry curvature

import matplotlib.pyplot as plt

# plot the bands
plt.subplot(211)
plt.scatter(k,e)
plt.ylabel("Energy")
plt.xlabel("k-path")
plt.xticks([])
# plot the Berry curvature
plt.subplot(212)
plt.plot(kb,b)
plt.ylabel("Berry curvature")
plt.xlabel("k-path")
plt.xticks([])


plt.show()






