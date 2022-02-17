# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import kdos
import numpy as np
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)

# create two Hamiltonians
h1 = h.copy()
h2 = h.copy()

# add opposite topological gaps
h1.add_haldane(0.1)
h2.add_haldane(-0.1)

# compute spectral function at the interface
kdos.interface(h1,h2,
               energies=np.linspace(-1.0,1.0,100), # energies
               nk=50, # number of kpoints
               delta=3e-2 # smearing
               )







