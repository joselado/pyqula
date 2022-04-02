# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
from pyqula import potentials
g = geometry.triangular_lattice() # create geometry
g = g.get_supercell([7,7]) # create a supercell
h = g.get_hamiltonian() # get the Hamiltonian
fmoire = potentials.commensurate_potential(g,n=3,minmax=[0,1]) # morie potential
h.add_onsite(fmoire) # add onsite energy following the moire
h.get_bands(operator=fmoire) # project on the moire




g.write_profile(fmoire)
