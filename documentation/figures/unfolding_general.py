import numpy as np
import style
from pyqula import geometry

# honeycomb lattice, non-diagonal supercell
g = geometry.honeycomb_lattice() # primitive geometry
M = [[2,1,0],[0,1,0],[0,0,1]] # non-diagonal supercell matrix, det(M)=2
gs = g.get_supercell(M,store_primal=True) # supercell, keeping the primitive cell info
h = gs.get_hamiltonian() # Hamiltonian of the supercell
kpath = gs.get_unfolded_kpath(nk=200) # primitive k-path, in supercell coordinates
(k,e,d) = h.get_kdos_bands(operator="unfold",delta=1e-1,kpath=kpath) # unfolded weight

# cubic lattice, 2 x 2 x 2 supercell
g = geometry.cubic_lattice() # three dimensional primitive geometry
gs = g.get_supercell([2,2,2],store_primal=True) # supercell, keeping the primitive cell info
h = gs.get_hamiltonian() # Hamiltonian of the supercell
kpath = gs.get_unfolded_kpath(nk=200) # primitive k-path, in supercell coordinates
(k3,e3,d3) = h.get_bands(operator="unfold",kpath=kpath) # unfolded band structure

nk, ne = len(np.unique(k)), len(np.unique(e)) # k is the outer index
(k,e,d) = [a.reshape(nk,ne) for a in (k,e,d)]
i = np.argsort(d3) # draw the states with weight on top
ticks = np.cumsum([0,.5,.5,np.sqrt(.5),np.sqrt(.75)]) # lengths of the segments of the cubic path

def draw(ax, c):
    im = ax[0].pcolormesh(k/k.max(), e, d, cmap=c.cmap, shading="auto", vmax=np.percentile(d,99))
    style.colorbar(im, ax[0], "unfolded weight")
    ax[0].set_xticks([0,1/3,1/2,2/3,1]) # the default path
    ax[0].set_xticklabels([r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"])
    ax[0].set_title(r"honeycomb, $\det M=2$")
    ax[1].scatter(k3/k3.max(), e3, s=8, color=c.ink2) # outline, so that zero weight stays visible
    sc = ax[1].scatter(k3[i]/k3.max(), e3[i], c=d3[i], s=4, cmap=c.cmap, vmin=0)
    style.colorbar(sc, ax[1], "unfolded weight")
    ax[1].set_xticks(ticks/ticks[-1])
    ax[1].set_xticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$", "R"])
    ax[1].set_title(r"cubic, $2\times2\times2$")
    for a in ax:
        a.set_xlim(0, 1)
        a.set_xlabel("momentum along the k-path")
        a.set_ylabel(r"energy $\omega$ [$t$]")

style.save(draw, "unfolding_general", npanels=2)
