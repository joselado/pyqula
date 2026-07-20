# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import films
from pyqula import meanfield

g = geometry.square_lattice()
h = g.get_hamiltonian()
h.add_onsite(-2.)
h.add_pairing(mode="dx2y2",delta=0.2)
energies = np.linspace(-2.0,2.0,100)
out = h.get_multi_fermi_surface(energies=energies,delta=4e-2,
                                 nk=200,
                                 nsuper=1)
dos = out[:,2,:].sum(axis=1) # total DOS at each energy

import matplotlib.pyplot as plt

plt.plot(energies,dos)
plt.xlabel("Energy")
plt.ylabel("DOS")
plt.show()
