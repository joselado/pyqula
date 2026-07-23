# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import matplotlib.pyplot as plt
import numpy as np

from pyqula import geometry
from pyqula import heterostructures

g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create the Hamiltonian
h1 = h.copy() # left superconducting lead
h2 = h.copy() # right superconducting lead
delta = 0.1 # superconducting gap
h1.add_swave(delta)
h2.add_swave(delta)

vs = np.linspace(0.02,1.5,40)*delta # bias voltages
for T in [0.1,0.5,1.0]: # loop over normal transparencies
    HT = heterostructures.build(h1,h2) # create the SNS junction
    HT.set_coupling(T) # set the normal transparency of the junction
    Is = HT.get_iv_curve(vs) # multiple-Andreev-reflection dc current

    plt.plot(vs/delta,Is,label="T=%.1f"%T)

plt.xlabel("Voltage [$\\Delta/e$]")
plt.ylabel("$I_{dc}$ [$e\\Delta/h$]")
plt.legend()
plt.show()
