# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
from pyqula import geometry
g = geometry.single_square_lattice() # create the basic geometry
h = geometry2hamiltonian(g,mw=0.0) # get the Hamiltonian, mw is the Wilson mass
from pyqula import scftypes
scf = scftypes.hubbardscf(h,U=1.,nkp=10,filling=0.5)
h = scf.hamiltonian
h.get_bands() # get the bandstructure
h.write()







