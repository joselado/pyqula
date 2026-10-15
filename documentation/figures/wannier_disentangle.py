import numpy as np
import style
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)

# one Wannier function out of both pz bands, with the deep part of the
# valence band frozen (kept exactly rather than optimized)
hwan = h.get_wannier_hamiltonian(bands=[0,1],num_wann=1,nk=12,
                                 dis_froz_max=-1.0,cutoff=0.0)

print("Wannier functions:",hwan.wannier_num_wann)
print("window used:",hwan.wannier_disentanglement_window)

(k,e) = h.get_bands(write=False) # the two bands of graphene
(kw,ew) = hwan.get_bands(write=False) # the band of the Wannier model

def draw(ax, c):
    ax.axhspan(-3.2, -1.0, color=c.green, alpha=0.15, lw=0, label="frozen window")
    ax.scatter(k, e, s=30, color=c.blue, label="graphene")
    ax.scatter(kw, ew, s=4, color=c.orange, label="Wannier model")
    ticks = np.array([0, 1/3, 1/2, 2/3, 1])*max(k) # the path goes through K, M and K'
    ax.set_xticks(ticks, [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"])
    ax.set_xlim(0, max(k)) ; ax.set_ylim(-3.2, 3.2)
    ax.set_ylabel("energy [$t$]")
    ax.legend(loc="upper center", markerscale=2)

style.save(draw, "wannier_disentangle")
