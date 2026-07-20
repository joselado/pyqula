# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import islands
import numpy as np
g = islands.get_geometry(name="honeycomb",n=6,nedges=3,rot=np.pi/6) # get an island
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian
from pyqula.potentialtk.vectorprofile import radial_vector_decay
fs = radial_vector_decay(v0=5.0,voo=1.0,rl=9.0,mode="exp")
h.add_strain(fs,mode="non_uniform")
#h.geometry.write_profile(f)
h.write_hopping()
(k,e) = h.get_bands()

import matplotlib.pyplot as plt
plt.scatter(k,e)
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.show()





