# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
import numpy as np
import matplotlib.pyplot as plt

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)

# real-space-impurity QPI: build a supercell, put a single strong
# onsite impurity at the origin, compute the real-space LDOS with
# ARPACK partial diagonalization (num_waves eigenstates around the
# target energy, efficient for large supercells) and Fourier transform
# it directly to get the QPI(q) signal -- see
# pyqula.qpitk.realspace.get_qpi_impurity for the full docstring
r,ldos_r,q,qpi_q = h.get_qpi_impurity(nsuper=10,
        impurities=[{"position": [0.,0.,0.], "onsite": 3.0}],
        energies=0.3, num_waves=60, nk=2, delta=0.2, write=False)

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))

s1 = ax1.scatter(r[:,0],r[:,1],c=ldos_r,cmap="inferno",s=15)
ax1.set_title("Real-space LDOS")
ax1.set_xlabel("x") ; ax1.set_ylabel("y") ; ax1.set_aspect("equal")
fig.colorbar(s1,ax=ax1)

# the q=0 point is just the total DOS (typically much larger than any
# scattering feature), so it is dropped here to keep the color scale
# sensitive to the actual QPI signal
away = np.sum(q**2,axis=1)>1e-8
s2 = ax2.scatter(q[away,0],q[away,1],c=qpi_q[away],cmap="inferno",s=60)
ax2.set_title("QPI(q) (Gamma point omitted)")
ax2.set_xlabel("qx") ; ax2.set_ylabel("qy") ; ax2.set_aspect("equal")
fig.colorbar(s2,ax=ax2)

plt.tight_layout()
plt.show()
