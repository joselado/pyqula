# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import specialhamiltonian # special Hamiltonians library
# TBG Hamiltonian
h = specialhamiltonian.twisted_bilayer_graphene(n=6,ti=0.4,has_spin=False) 
h.set_filling(0.5,nk=1)
(k,e) = h.get_bands(num_bands=20,kpath=["G","K","M","K'","G"]) # computebands

import matplotlib.pyplot as plt
plt.scatter(k,e)
plt.xlabel("Momentum") ; plt.ylabel("Energy") ; plt.xticks([])
plt.show()







