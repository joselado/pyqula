import numpy as np
import style
from pyqula import geometry
from pyqula import potentials
g0 = geometry.triangular_lattice() # primitive geometry
g = g0.get_supercell(12,store_primal=True) # supercell of 144 primitive cells
h = g.get_hamiltonian() # Hamiltonian of the supercell
fmoire = potentials.commensurate_potential(g,n=3,minmax=[0,1]) # moire potential
h.add_onsite(fmoire) # onsite energy following the moire
kpath = g.get_unfolded_kpath(nk=100) # primitive k-path, in supercell coordinates
(k,e,d) = h.get_kdos_bands(operator="unfold",mode="KPM",delta=5e-2,kpath=kpath,
        energies=np.linspace(-3,-1,200)) # unfolded weight, with the KPM

nk, ne = len(np.unique(k)), len(np.unique(e)) # k is the outer index
(k,e,d) = [a.reshape(nk,ne) for a in (k,e,d)]

def draw(ax, c):
    im = ax.pcolormesh(k/k.max(), e, d, cmap=c.cmap, shading="auto", vmax=np.percentile(d,99))
    style.colorbar(im, ax, r"unfolded weight $A(\mathbf{k},\omega)$")
    ax.set_xticks([0,1/3,1/2,2/3,1]) # the default path
    ax.set_xticklabels([r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"])
    ax.set_xlabel("momentum along the k-path")
    ax.set_ylabel(r"energy $\omega$ [$t$]")

style.save(draw, "unfolding_kpm")
