# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
import numpy as np
import matplotlib.pyplot as plt

g = geometry.square_lattice()
h = g.get_hamiltonian(has_spin=False)

# QPI of a single point impurity in the infinite square lattice, from the
# T-matrix T = V [1 - G0 V]^-1 with G0 the local Green's function of the
# clean system. The impurity is an onsite potential of 5t, a strong one
(qs,es,qpi) = h.get_qpi(mode="tmatrix",impurity=5.0,nk=120,nsuper=2,
        delta=1e-1,energies=[-1.0],write=False)

# the same impurity in the Born approximation, T = V
(qs,es,qpib) = h.get_qpi(mode="born",impurity=5.0,nk=120,nsuper=2,
        delta=1e-1,energies=[-1.0],write=False)

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
s1 = ax1.scatter(qs[:,0],qs[:,1],c=qpi[0],cmap="inferno",s=2)
ax1.set_title("T-matrix QPI")
ax1.set_xlabel("qx [2pi/a]") ; ax1.set_ylabel("qy [2pi/a]")
ax1.set_aspect("equal") ; fig.colorbar(s1,ax=ax1)
s2 = ax2.scatter(qs[:,0],qs[:,1],c=qpib[0],cmap="inferno",s=2)
ax2.set_title("Born QPI")
ax2.set_xlabel("qx [2pi/a]") ; ax2.set_ylabel("qy [2pi/a]")
ax2.set_aspect("equal") ; fig.colorbar(s2,ax=ax2)
plt.tight_layout()
plt.show()
