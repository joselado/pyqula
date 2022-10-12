# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

n = 3 # size of the unit cell
ti = 0.3 # interlayer hopping

from pyqula import specialhamiltonian # special Hamiltonians library
h = specialhamiltonian.twisted_bilayer_graphene(n=n,ti=ti) # TBG Hamiltonian
h.set_filling(0.5,nk=2) # first mode
(k1,e1) = h.get_bands(num_bands=20) # compute band structure


from pyqula import specialgeometry 
from pyqula import specialhopping
g = specialgeometry.twisted_bilayer(n)
h = g.get_hamiltonian(mgenerator=specialhopping.twisted_matrix(ti=ti))
h.set_filling(0.5,nk=2) # first mode
(k2,e2) = h.get_bands(num_bands=20) # compute band structure

import matplotlib.pyplot as plt
plt.scatter(k2,e2,s=30,c="blue",label="specialhamiltonian")
plt.scatter(k1,e1,s=10,c="red",label="specialgeometry")
plt.xticks([])
plt.legend()
plt.show()



