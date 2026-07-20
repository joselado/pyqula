# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import specialhamiltonian
h = specialhamiltonian.multilayer_graphene(l=[-1,0,1])
(k,e) = h.get_bands()

import matplotlib.pyplot as plt

plt.scatter(k,e)
plt.xlabel("k-path") ; plt.ylabel("Energy")
plt.show()
