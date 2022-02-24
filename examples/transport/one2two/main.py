# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import heterostructures
import numpy as np
import matplotlib.pyplot as plt
g1 = geometry.chain()
g2 = geometry.chain()
g2 = geometry.single_square_lattice()
h1 = g1.get_hamiltonian(has_spin=False)
h1 = h1*10
h2 = g2.get_hamiltonian(has_spin=False)
h2.get_dos() ; exit()
ht = heterostructures.build(h1,h2) # create the junction
ht.delta = 1e-3 # imaginary part of the energy
ht.scale_rc = 1e-1
es = np.linspace(-5.0,5.0,40) # grid of energies
#ts = [ht.didv(energy=e) for e in es] # calculate transmission
ts = [ht.didv(energy=e) for e in es] # calculate transmission
print(ts)
plt.plot(es,ts,marker="o")
plt.ylim([0,4.1])
plt.show()







