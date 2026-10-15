import numpy as np
import style
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.add_onsite([0.8,-0.8]) # sublattice potential opens a gap

# Wannierize just the lowest (valence) band: bands=[0,0] selects band
# index 0 at every k-point of a 12x12 Monkhorst-Pack wannierization mesh
hwan = h.get_wannier_hamiltonian(bands=[0,0],nk=12)

(k,e,sl) = h.get_bands(operator="sublattice",write=False) # colored by sublattice
(kw,ew) = hwan.get_bands(write=False)

def draw(ax, c):
    sc = ax.scatter(k, e, c=sl, s=10, cmap=c.diverging, vmin=-1, vmax=1)
    ax.plot(kw, ew, "--", color=c.ink, lw=1.4, label="Wannier model")
    style.colorbar(sc, ax, r"sublattice $\langle \sigma_z\rangle$")
    ticks = np.array([0, 1/3, 1/2, 2/3, 1])*max(k) # the path goes through K, M and K'
    ax.set_xticks(ticks, [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"])
    ax.set_xlim(0, max(k))
    ax.set_ylabel("energy [$t$]")
    ax.legend(loc="center right")

style.save(draw, "wannier_valence")
