# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# An s-wave superconductor, the spinful chain with an attractive Hubbard U=-2t
# at filling 0.3, under a slow onsite potential with one period over n cells.
# The pairing follows the potential. With the Nambu degree of freedom the
# ultracell keeps the Bogoliubov-de Gennes states with |E|<=emax, counted from
# the chemical potential, and freezes the deep ones with their pairing. So a
# uniform pairing is exact for any emax. The modulated part converges slowly
# with emax, because the frozen states do not respond to the potential, and
# we raise emax until the result stops changing. With every state kept the
# pairing is 0.382 on average and its modulation 0.032, the same as "ed" on
# the supercell. With emax=3 the ultracell keeps 212 of 240 states and the
# modulation is 0.036, with emax=2 it keeps 156 and the modulation is 0.047,
# while the mean pairing moves by 0.005 only. The cut exaggerates the
# modulation, and it converges from above as emax grows.

import numpy as np
from pyqula import geometry
h = geometry.chain().get_hamiltonian() # spinful chain, hopping t=1
n = 60 # period of the potential, in unit cells
hs = h.get_supercell(n) # the ultracell, written as a supercell
hs.add_onsite(lambda r: 0.4*np.cos(2*np.pi*r[0]/n)) # slow potential
h.setup_nambu_spinor() ; hs.setup_nambu_spinor() # electrons and holes
kw = dict(U=-2.,filling=0.3,mf="swave",nk=4,maxerror=1e-8) # nk=4 is converged
x = np.array(hs.geometry.r)[:,0] # position along the chain
ds = [] # pairing on each site
for emax in [None,3.,2.]: # None keeps every state
    u = h.get_ultracell(hs,emax=emax)
    # a cut on a Nambu ultracell needs the basis rebuilt at every iteration
    u2 = u.get_mean_field_hamiltonian(rebuild_basis=True,**kw)
    size = u2.get_hk_gen()(np.array([0.1,0.,0.])).shape[0] # states kept
    d = np.abs(u2.hs.extract("swave")) # pairing on each site
    ds.append(d)
    print("emax",emax,"states kept",size,"mean pairing",np.mean(d),
            "modulation",(np.max(d)-np.min(d))/2)

import matplotlib.pyplot as plt
for d,label in zip(ds,["every state","emax=3","emax=2"]):
    plt.plot(x,d,"o",label=label)
plt.xlabel("position [unit cells]") ; plt.ylabel("s-wave pairing")
plt.legend() ; plt.show()
