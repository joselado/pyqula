# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Z2 invariant of a time-reversal-symmetric (class DIII) superconducting
# wire: helical p-wave pairing with Rashba spin-orbit coupling, driven
# trivial by a competing s-wave pairing, and the bulk gap that closes at the
# transition
import numpy as np
from pyqula import geometry

dss = np.linspace(0.,1.6,17) # s-wave pairing
nus,gaps = [],[]
for ds in dss:
    h = geometry.chain().get_hamiltonian() # spinful chain
    h.add_onsite(0.5) # shift the band, keeping the Fermi energy inside it
    h.add_rashba(0.2) # Rashba spin-orbit coupling
    h.add_pairing(mode="pwave",delta=0.5j,d=[1.,0.,0.]) # helical p-wave
    if ds>0.: h.add_swave(ds) # competing s-wave pairing
    nus.append(h.get_topological_invariant()) # Z2 invariant
    (k,e) = h.get_bands(nk=400) # BdG bands
    gaps.append(np.min(np.abs(e))) # bulk gap
    print("s-wave =",round(ds,2),"  Z2 =",nus[-1],"  gap =",round(gaps[-1],3))

import matplotlib.pyplot as plt
plt.plot(dss,nus,marker="o",label="Z2 invariant")
plt.plot(dss,gaps,label="bulk gap")
plt.xlabel("s-wave pairing") ; plt.legend()
plt.show()
