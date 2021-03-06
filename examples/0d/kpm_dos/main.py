# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry  # library to create crystal geometries
import numpy as np
g = geometry.chain()
g = g.get_supercell(3000) # big supercell
g.dimensionality = 0 # make it zero dimensional
h = g.get_hamiltonian(is_sparse=True,has_spin=False) # in sparse mode
(x,y) = h.get_dos(mode="KPM",
            energies=np.linspace(-3.0,3.0,200), # energies
            delta=1e-4, # smearing
            ntries=10 # number of vectors for stochastic trace
            )

