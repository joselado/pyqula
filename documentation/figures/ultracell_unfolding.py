import numpy as np
import style
from pyqula import geometry

g = geometry.honeycomb_lattice() # unit cell of graphene
h = g.get_hamiltonian() # first-neighbor hopping, t=1
n = 20 # period of the modulation, in unit cells along a1
hs = h.get_supercell([n,1]) # the ultracell, written as a supercell
b1 = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T)[0] # b1.r is the coordinate along a1
V0 = 0.3 # amplitude of the potential, in units of t
hs.add_onsite(lambda r: V0*np.cos(2*np.pi*b1.dot(r)/n)) # one period over n cells
u = h.get_ultracell(hs) # hs solved in the Bloch states of graphene
(k,e,a) = u.get_kdos_bands(energies=np.linspace(-1.5,1.5,301),nk=400,delta=0.02)

nk, ne = len(np.unique(k)), len(np.unique(e)) # k is the outer index
(k,e,a) = [x.reshape(nk,ne) for x in (k,e,a)]
uc = h.get_ultracell(h.get_supercell([n,1])) # the same without the potential
a0 = uc.get_kdos_bands(energies=np.linspace(-1.5,1.5,301),nk=400,delta=0.02)[2]
print("largest change of the spectral function:",np.max(np.abs(a-a0.reshape(nk,ne))))
print("total weight on the path, with and without the potential:",a.sum(),a0.sum())

def draw(ax, c):
    im = ax.pcolormesh(k, e, a, cmap=c.cmap, shading="auto", vmax=np.percentile(a,99))
    style.colorbar(im, ax, r"spectral function $A(k,\omega)$")
    style.kpath(ax, k, [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"], at=[0,1/3,1/2,2/3,1])
    ax.set_xlabel("momentum along the k-path")
    ax.set_ylabel(r"energy $\omega$ [$t$]")

style.save(draw, "ultracell_unfolding")
