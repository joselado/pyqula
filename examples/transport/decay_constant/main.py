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
h.shift_fermi(1.) # shift the chemical potential
h1 = h.copy() # copy
h2 = h.copy() # copy
h1.add_swave(.0) # add electron hole symmetry
h2.add_swave(.1) # pairing gap of 0.01
#h2.add_pairing(mode="triplet",delta=0.05) # pairing gap of 0.01
ht = heterostructures.build(h1,h2) # create the junction
ht.delta = 1e-8 # analytic continuation of the Green's functions
es = np.linspace(-.2,.2,101) # grid of energies
T = 2e-2 # reference transparency 
ht.scale_lc = T # set the transparency for dIdV
ts = [ht.didv(energy=e) for e in es] # calculate transmission
ks = [ht.get_kappa(energy=e,T=T) for e in es] # calculate transmission
plt.subplot(121)
plt.plot(es,ts,marker="o")
plt.subplot(122)
plt.plot(es,ks,marker="o")
plt.ylim([0,4.1])
plt.show()







