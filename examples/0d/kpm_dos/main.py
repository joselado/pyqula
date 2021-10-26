# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry  # library to create crystal geometries
import numpy as np
g = geometry.chain()
g = g.supercell(10000) # big supercell
g.dimensionality = 0
h = g.get_hamiltonian(is_sparse=True,has_spin=False) # in sparse mode
(x,y) = h.get_dos(mode="KPM",
            energies=np.linspace(-3.0,3.0,200), # energies
            delta=1e-2, # smearing
            ntries=100 # number of vectors for stochastic trace
            )

