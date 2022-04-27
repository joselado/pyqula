# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
import matplotlib.pyplot as plt

# this example performs a selconsistent calculation in a dimer

g = geometry.chain() # generate the geometry
g = g.supercell(2)
g.dimensionality = 0
print(g.r)
for phi in np.linspace(0.,1.,100):
    h = g.get_hamiltonian()
    h.add_swave(lambda r: (r[0]>0.)*0.2)
    h.add_swave(lambda r: (r[0]<0.)*0.2*np.exp(1j*phi*np.pi*2.))
    ks,es = h.get_bands()
    plt.scatter(es*0.+phi,es,c="black")

plt.xlabel("phase difference")
plt.ylabel("Energy")
plt.show()
