import numpy as np
import style
from pyqula import geometry

g = geometry.honeycomb_lattice() # unit cell of graphene
h = g.get_hamiltonian(has_spin=False) # first-neighbor hopping, t=1
M = [[2,1,0],[-1,1,0],[0,0,1]] # the sqrt3 x sqrt3 cell, A_S = M A
hs = h.get_supercell(M) # the ultracell, written as a supercell
hs.add_kekule(0.1) # the Kekule modulation of the hoppings
u = h.get_ultracell(hs) # hs solved in the Bloch states of graphene
(k,e,a) = u.get_kdos_bands(energies=np.linspace(-1.,1.,301),nk=400,delta=0.01)

nk, ne = len(np.unique(k)), len(np.unique(e)) # k is the outer index
(k,e,a) = [x.reshape(nk,ne) for x in (k,e,a)]
for name,p in [("K",[1/3,1/3,0.]),("K'",[2/3,2/3,0.])]:
    es = np.linalg.eigvalsh(u.get_hk_gen()(u.basis.M@np.array(p)))
    print("levels closest to zero at",name,":",np.sort(es[np.argsort(np.abs(es))[0:2]]))

def draw(ax, c):
    im = ax.pcolormesh(k, e, a, cmap=c.cmap, shading="auto", vmax=np.percentile(a,99))
    style.colorbar(im, ax, r"spectral function $A(k,\omega)$")
    style.kpath(ax, k, [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"], at=[0,1/3,1/2,2/3,1])
    ax.set_xlabel("momentum along the k-path")
    ax.set_ylabel(r"energy $\omega$ [$t$]")

style.save(draw, "ultracell_kekule")
