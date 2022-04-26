# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import specialhamiltonian # special Hamiltonians library
h = specialhamiltonian.twisted_bilayer_graphene(13,ti=0.4) # TBG Hamiltonian
h.set_filling(.5,nk=1) # put Fermi energy at half filling
h.turn_sparse()
from pyqula import parallel
parallel.cores = 6
(k,e) = h.get_bands(num_bands=20,nk=100) # compute band structure

