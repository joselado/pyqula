import numpy as np
import style
from pyqula import specialhamiltonian
h = specialhamiltonian.three_band_tmdc("MoS2") # MoS2, TNN model, GGA, with SOC
(k,e,sz) = h.get_bands(operator="sz") # bands, and the spin of each state
print("gap in eV:",h.get_gap())

def draw(ax, c):
    sc = ax.scatter(k, e, c=sz, s=6, cmap=c.diverging, vmin=-1, vmax=1)
    style.colorbar(sc, ax, r"spin $\langle S_z\rangle$")
    style.kpath(ax, k, [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"],
                at=[0,1/3,1/2,2/3,1])
    ax.axhline(0, color=c.ink, lw=0.6, alpha=0.4)
    ax.set_xlabel("momentum")
    ax.set_ylabel("energy [eV]")

style.save(draw, "hamiltonian_tmdc")
