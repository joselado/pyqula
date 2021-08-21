# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import heterostructures
import numpy as np
import matplotlib.pyplot as plt
g = geometry.square_ribbon(1)
#g = g.supercell(3)
h = g.get_hamiltonian()
h.shift_fermi(1.0) # shift the chemical potential
h1 = h.copy() # copy
h2 = h.copy() # copy
h1.add_swave(.0) # add electron hole symmetry
h2.add_swave(.1) # pairing gap of 0.01
#h2.add_pairing(mode="triplet",delta=0.05) # pairing gap of 0.01
ht = heterostructures.build(h2,h1) # create the junction
ht.delta = 1e-12 # analytic continuation of the Green's functions

ts = np.linspace(1e-3,1.0,30)
gs = [] # empty list
e = 0.07
for t in ts:
    ht.scale_rc = t # set the transparency for dIdV
    g = ht.didv(energy=e)
    gs.append(g)

plt.subplot(121)
#plt.plot(es,ts,marker="o")
plt.subplot(122)
plt.plot(np.log(ts),np.log(gs),marker="o")
plt.show()







