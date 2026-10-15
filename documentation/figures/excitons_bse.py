import numpy as np
import style
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction
g = geometry.honeycomb_lattice()
h2 = g.get_hamiltonian()
h2.add_sublattice_imbalance(1.0) # a gapped semiconductor
W = density_interaction(h2,Vr=lambda r1,r2: 0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25)) # Coulomb tail
bse = h2.get_bse(V=W,nk=21)
print(bse.get_energies()[0], bse.get_binding_energies()[0]) # lowest exciton, and how far below the gap

es = bse.get_energies() # all the exciton energies
de = bse.pairs.dE # energies of the free electron-hole pairs
lowest = np.isclose(es, es[0]) # the degenerate states of the lowest exciton
weight = np.zeros(len(bse.pairs.kpoints)) # |A(k)|^2 of the lowest exciton
for A in bse.amplitudes[lowest]:
    np.add.at(weight, bse.pairs.kindex, np.abs(A)**2)
weight = weight/np.sum(lowest) # average over the degenerate states
k1 = np.arange(21)/21 # the mesh, in units of the reciprocal lattice vectors

def draw(ax, c):
    for (e, color, label) in [(de, c.ink2, "free pairs"), (es, c.blue, "excitons")]:
        (levels, counts) = np.unique(e.round(4), return_counts=True)
        ax[0].vlines(levels, 0, counts, color=color, lw=2, label=label)
    ax[0].set_xlim(1.8, 2.25)
    ax[0].set_ylim(0, 52)
    ax[0].set_xlabel(r"energy [$t$]")
    ax[0].set_ylabel("number of states")
    ax[0].legend(loc="upper left")
    im = ax[1].pcolormesh(k1, k1, weight.reshape(21,21).T, cmap=c.cmap,
                          shading="nearest")
    style.colorbar(im, ax[1], r"weight $\sum_{vc}|A_{vc}(k)|^2$")
    ax[1].scatter([1/3, 2/3], [1/3, 2/3], marker="x", color=c.blue, s=60)
    ax[1].text(1/3+0.04, 1/3-0.04, "K", color=c.blue, va="top")
    ax[1].text(2/3+0.04, 2/3-0.04, "K'", color=c.blue, va="top")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("$k_1$")
    ax[1].set_ylabel("$k_2$")

style.save(draw, "excitons_bse", npanels=2)
