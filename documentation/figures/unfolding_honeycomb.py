import style
from pyqula import geometry
import numpy as np
g = geometry.honeycomb_lattice() # primitive geometry
n = 3
gs = g.get_supercell(n,store_primal=True) # supercell, keeping the primitive cell info
h = gs.get_hamiltonian() # Hamiltonian of the supercell
kpath = gs.get_unfolded_kpath(nk=200) # primitive k-path, in supercell coordinates
(k,e,d) = h.get_kdos_bands(operator="unfold",delta=1e-1,kpath=kpath) # unfolded weight
(k0,e0) = h.get_bands(kpath=kpath) # the folded bands, on the same path

nk, ne = len(np.unique(k)), len(np.unique(e)) # k is the outer index
(k,e,d) = [a.reshape(nk,ne) for a in (k,e,d)]

def draw(ax, c):
    ax[0].scatter(k0/k0.max(), e0, s=1, color=c.blue)
    ax[0].set_title("bands of the supercell")
    im = ax[1].pcolormesh(k/k.max(), e, d, cmap=c.cmap, shading="auto", vmax=np.percentile(d,99))
    style.colorbar(im, ax[1], r"unfolded weight $A(\mathbf{k},\omega)$")
    ax[1].set_title("unfolded")
    for a in ax:
        a.set_xticks([0,1/3,1/2,2/3,1]) # the default path
        a.set_xticklabels([r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"])
        a.set_xlim(0, 1)
        a.set_ylim(e.min(), e.max())
        a.set_xlabel("momentum along the k-path")
        a.set_ylabel(r"energy $\omega$ [$t$]")

style.save(draw, "unfolding_honeycomb", npanels=2)
