# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import matplotlib.pyplot as plt


from pyqula import geometry
from pyqula import heterostructures
import numpy as np

g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create teh Hamiltonian
h.add_onsite(1.0)
h1 = h.copy() # first lead
h2 = h.copy() # second lead
delta = 0.01 # superconducting gap
h2.add_swave(delta) # the second lead is superconducting
es = np.linspace(-.03,.03,100) # grid of energies
HT = heterostructures.build(h1,h2) # create the junction
T = 0.3
HT.set_coupling(T) # set the coupling between the leads 
# get the total conductance, the electron one, and Andreev reflection
G = [HT.didv(energy=e) for e in es] # calculate transmission
Ge = [HT.didv(energy=e,component="electron") for e in es] 
Gh = [HT.didv(energy=e,component="Andreev") for e in es]

# first total conductance
plt.subplot(1,2,1)
plt.plot(es/delta,G,label="total",linewidth=6)
plt.legend()
plt.xlabel("Energy [$\\Delta$]")
plt.ylabel("dIdV [arb.units.]")
plt.xlim([-3,3])

# now each component
plt.subplot(1,2,2)
plt.plot(es/delta,Ge,label="electron")
plt.plot(es/delta,Gh,label="Andreev")
plt.legend()
plt.xlabel("Energy [$\\Delta$]")
plt.ylabel("dIdV [arb.units.]")
plt.xlim([-3,3])

plt.tight_layout()
plt.show()







