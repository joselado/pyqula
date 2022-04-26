# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import specialhamiltonian # special Hamiltonians library
h = specialhamiltonian.twisted_bilayer_graphene(n=4,ti=lambda r: 0.4) # TBG Hamiltonian
# positive value of ti would yield a negative interlayer hopping
h.set_filling(0.5,nk=1)
(k,e) = h.get_bands(num_bands=10) # compute band structure








