# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import heterostructures
import matplotlib.pyplot as plt
import numpy as np
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
# copy Hamiltonian
h1 = h.copy()
h2 = h.copy()
hfun = heterostructures.build(h1,h2) # build the heterostructure
hfun.delta = 0.01 # imaginary part
es = np.linspace(-.2,.2,21) # grid of energies
# for a 2d system, the conductance is obtained by summing over k parallel
ts = [hfun.didv(energy=e,error=0.1,nk=500) for e in es] # calculate the transmissions
plt.plot(es,ts,marker="o") # plot result
plt.show()







