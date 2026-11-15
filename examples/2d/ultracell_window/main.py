# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# The insulator of examples/2d/ultracell_meanfield: the honeycomb lattice with
# onsite energies +1 on A and -1 on B, under a slow exchange field with one
# period over n unit cells, with a Hubbard U. Here the ultracell keeps only the
# states of the unit cell with energy between emin and emax, window=(emin,emax).
# The states below emin are frozen: they stay full with the density of the
# unit cell, so they do not respond to the field. The states above emax are
# dropped. So a window makes the matrix of the ultracell smaller, and what it
# misses is the response of the deep states. With the window (-2,2) the
# ultracell keeps 40 of its 96 states, and the induced moment is off by 6
# percent of its largest value, the part the deep states give. The window
# (-1.5,1.5) keeps 24 states and is 42 percent off, since it freezes most of
# the valence band. You check a window by widening it until the moment stops
# changing.

import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice() # unit cell of the honeycomb lattice
h = g.get_hamiltonian() # spinful, first-neighbor hopping t=1
h.add_sublattice_imbalance(1.0) # +1 on A and -1 on B, a gap of 2t
n = 24 # period of the modulation, in unit cells along a1
hs = h.get_supercell([n,1]) # the ultracell, written as a supercell
b1 = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T)[0] # b1.r is the coordinate along a1
hs.add_exchange(lambda r: [0.,0.,0.05*np.cos(2*np.pi*b1.dot(r)/n)]) # along z
nk = [2,12] # k-mesh of the ultracell, 24 cells long along a1
x = np.array([b1.dot(r) for r in hs.geometry.r]) # coordinate along a1
ms = [] # induced moments
for window in [None,(-2.,2.),(-1.5,1.5)]: # None keeps every state
    u = h.get_ultracell(hs,window=window)
    size = u.get_hk_gen()(np.array([0.1,0.2,0.])).shape[0] # states kept
    u2 = u.get_mean_field_hamiltonian(U=1.5,filling=0.5,mf="ferroZ",nk=nk,
            maxerror=1e-8)
    m = u2.get_vev(operator="sz",nk=nk) # induced moment on each site
    ms.append(m)
    print("window",window,"states kept",size,"largest error of the moment",
            np.max(np.abs(m-ms[0]))/np.max(np.abs(ms[0])))

import matplotlib.pyplot as plt
o = np.argsort(x)
for m,label in zip(ms,["every state","window (-2,2)","window (-1.5,1.5)"]):
    plt.plot(x[o],m[o],"o",label=label)
plt.xlabel("position along a1 [unit cells]") ; plt.ylabel("induced moment")
plt.legend() ; plt.show()
