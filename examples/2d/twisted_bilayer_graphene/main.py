# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import specialhamiltonian # special Hamiltonians library
h = specialhamiltonian.twisted_bilayer_graphene(n=6,ti=0.4) # TBG Hamiltonian
h.set_filling(0.5,nk=1)
(k,e) = h.get_bands(num_bands=20) # compute band structure








