import style
from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice() # primitive geometry
gs = g.get_supercell(np.sqrt(3),store_primal=True) # sqrt(3) x sqrt(3) supercell
h = gs.get_hamiltonian() # Hamiltonian of the supercell
fons = lambda r: (np.sum((r - gs.r[0])**2)<1e-2)*0.6 # onsite in one of the three sites
h.add_onsite(fons) # three-sublattice charge order
kpath = gs.get_unfolded_kpath(nk=200) # primitive k-path, in supercell coordinates
(k,e,d) = h.get_kdos_bands(operator="unfold",delta=1e-1,kpath=kpath,
        energies=np.linspace(-4,3,300)) # unfolded weight, down to below the band

nk, ne = len(np.unique(k)), len(np.unique(e)) # k is the outer index
(k,e,d) = [a.reshape(nk,ne) for a in (k,e,d)]

def draw(ax, c):
    im = ax.pcolormesh(k/k.max(), e, d, cmap=c.cmap, shading="auto",
            norm="log", vmin=0.3, vmax=d.max()) # a log scale shows the weak shadow band
    style.colorbar(im, ax, r"unfolded weight $A(\mathbf{k},\omega)$")
    ax.set_xticks([0,1/3,1/2,2/3,1]) # the default path
    ax.set_xticklabels([r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"])
    ax.set_xlabel("momentum along the k-path")
    ax.set_ylabel(r"energy $\omega$ [$t$]")

style.save(draw, "unfolding_sqrt3")
