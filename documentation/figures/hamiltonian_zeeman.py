import numpy as np
import style
from pyqula import geometry

g = geometry.chain() # geometry of the 1D chain
h = g.get_hamiltonian() # generate the Hamiltonian
h.add_zeeman([0.,0.,0.3]) # Zeeman field along z
(k,e,sz) = h.get_bands(operator="sz") # bands, and the spin of each state
k = 2*np.pi*k/k.max() # the k-path goes from 0 to 2 pi

def draw(ax, c):
    sc = ax.scatter(k, e, c=sz, s=8, cmap=c.diverging, vmin=-1, vmax=1)
    style.colorbar(sc, ax, r"spin $\langle S_z\rangle$")
    style.kpath(ax, k, ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel("momentum $k$")
    ax.set_ylabel("energy [$t$]")

style.save(draw, "hamiltonian_zeeman")
