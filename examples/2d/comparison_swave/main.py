# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import matplotlib.pyplot as plt
from pyqula import geometry


g = geometry.square_lattice()
h = g.get_hamiltonian()
h.add_onsite(-3.0)
h.add_swave(0.9)
(k,e,c) = h.get_bands(operator='electron', nk=100)
plt.subplot(1, 2, 1)
plt.scatter(k,e,c=c, s=200*c)
plt.title('swave')


g = geometry.square_lattice()
h = g.get_hamiltonian()
h.add_onsite(-3.0)
h.add_pairing(mode='swave',delta=0.9,nn=0)
(k,e,c) = h.get_bands(operator='electron', nk=100)
plt.subplot(1,2,2)
plt.scatter(k,e,c=c, s=200*c)
plt.title('add_pairing')





plt.show()
