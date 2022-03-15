# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import matplotlib.pyplot as plt


from pyqula import geometry
from pyqula import heterostructures
import numpy as np

g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create teh Hamiltonian
h1 = h.copy() # first lead
h2 = h.copy() # second lead
h2.add_swave(.01) # the second lead is superconducting
es = np.linspace(-.03,.03,100) # grid of energies
for T in np.linspace(1e-3,1.0,6): # loop over transparencies
    HT = heterostructures.build(h1,h2) # create the junction
    HT.set_coupling(T) # set the coupling between the leads 
    Gs = [HT.didv(energy=e) for e in es] # calculate transmission




    plt.plot(es/0.01,Gs/Gs[0])


plt.ylim([0,4.1])
plt.xlim([-3,3])
plt.yticks([])
plt.xlabel("Energy [$\\Delta$]")
plt.ylabel("dIdV [arb.units.]")
plt.show()







