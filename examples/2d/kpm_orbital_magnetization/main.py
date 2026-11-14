# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Orbital magnetization of a Haldane flake as a function of the Fermi level,
# from a Chebyshev expansion of the Fermi function. The moment is
# M = Tr[f(H) m_z], with m_z = -(r x v)_z/2 the moment of the current of the
# electrons. Inside the bulk gap only the edge states change with E_F, and
# the slope dM/dE_F is C A/(2 pi), with C = 1 the Chern number and A the area
# of the flake (the Streda relation). The edge current flows a little inside
# the edge, so on 600 sites the slope is 0.91 of C, and closer to 1 on a
# larger flake.
import numpy as np
from pyqula import islands

g = islands.get_geometry(name="honeycomb",n=10,nedges=6) # 600 sites
h = g.get_hamiltonian(has_spin=False,is_sparse=True)
h.add_haldane(0.2) # Chern insulator, C=1, bulk gap of 1.8
efs = np.linspace(-2.,2.,81) # Fermi levels
m = h.get_orbital_magnetization(fermi=efs,T=0.02,trace="exact") # total moment
area = len(g.r)*3*np.sqrt(3)/4 # area of the flake
gap = np.abs(efs)<0.4 # Fermi levels inside the bulk gap
slope = np.polyfit(efs[gap],m[gap],1)[0] # dM/dE_F
print("2 pi (dM/dE_F)/A in the gap =",round(2*np.pi*slope/area,3)) # about 0.91

import matplotlib.pyplot as plt
plt.plot(efs,m/area)
plt.xlabel("$E_F$") ; plt.ylabel("$M_z/A$")
plt.show()
