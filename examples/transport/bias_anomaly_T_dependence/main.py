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
#h2.add_swave(.1) # pairing gap of 0.01
h2.add_pairing(mode="triplet",delta=0.05,d=[0.,0.,1.]) # pairing gap of 0.01
ht = heterostructures.build(h1,h2) # create the junction
ht.delta = 1e-6 # analytic continuation of the Green's functions
temps = np.linspace(0.,.01,20) # grid of energies
T = 1e-1 # reference transparency 
ht.scale_lc = T # set the transparency for dIdV
from pyqula import parallel
parallel.cores = 6
ts = parallel.pcall(lambda temp: ht.didv(energy=0.0,temp=temp),temps)
plt.plot(temps,ts,marker="o")
plt.show()







