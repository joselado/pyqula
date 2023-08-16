# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula.heterostructures import LocalProbe
import numpy as np
import matplotlib.pyplot as plt
g = geometry.chain()
h = g.get_hamiltonian()
h.shift_fermi(1.) # shift the chemical potential
D = 0.1 # superconducting gap
h.add_swave(D) # pairing gap of 0.1
lp = LocalProbe(h,delta=1e-8) # create a local probe object
lp.T = 2e-2 # reference transparency 
es = np.linspace(-2*D,2*D,101) # grid of energies
ts = [lp.didv(energy=e) for e in es] # calculate transmission
ks = [lp.get_kappa(energy=e) for e in es] # calculate decay rate
plt.subplot(121)
plt.plot(es/D,ts,marker="o")
plt.xlabel("Energy/$\Delta$") ; plt.ylabel("dIdV") ; plt.ylim([0.,0.006])
plt.subplot(122)
plt.plot(es/D,ks,marker="o")
plt.xlabel("Energy/$\Delta$") ; plt.ylabel("$\\kappa/\\kappa_N$")
plt.ylim([0,4.1])
plt.tight_layout()
plt.show()







