# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import specialgeometry # special geometry
g = specialgeometry.tbg(3) # generate geometry
s = 0.03 # amount of strain
g.add_strain(s) # add strain to the geometry
from pyqula import specialhamiltonian # special Hamiltonians library
# generate the Hamiltonian using as input the strained geometry
h = specialhamiltonian.twisted_bilayer_graphene(g=g,ti=0.4) # TBG Hamiltonian
(k,e) = h.get_bands(num_bands=20) # compute band structure








