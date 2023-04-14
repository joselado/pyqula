# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import embedding
import numpy as np
g = geometry.chain() # create geometry of a chain
g = g.get_supercell(10)
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian,spinless

eb = embedding.Embedding(h,selfenergy=1.) # create the embedding object
es = np.linspace(-3.0,3.0,400) # energies
ds = [eb.get_dos(energy=e,delta=1e-3) for e in es]

import matplotlib.pyplot as plt
plt.plot(es,ds)
plt.show()
