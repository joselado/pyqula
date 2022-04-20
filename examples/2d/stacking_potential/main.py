# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import specialgeometry # special Hamiltonians library
g = specialgeometry.tbg(6) # TBG Hamiltonian
from pyqula import potentials
f = potentials.tbgAA(g) # get a profile distinguishing AA from AB
g.write_profile(f) # write the profile in a file







