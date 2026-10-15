import numpy as np
import style
from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
sx = h.get_operator("sx") # Spin x component
sy = h.get_operator("sy") # Spin y component
sz = h.get_operator("sz") # Spin z component
h.add_rashba(0.3) # spin-orbit coupling, rotates the spin along the band
h.add_zeeman([0.,0.,0.3]) # Zeeman field along z
(k,e,s) = h.get_bands(operator=sz) # bands, and S_z of each state

def draw(ax, c):
    sc = ax.scatter(k, e, c=s, s=6, cmap=c.diverging, vmin=-1, vmax=1)
    style.colorbar(sc, ax, r"spin $\langle S_z\rangle$")
    style.kpath(ax, k, [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"], at=[0,1/3,1/2,2/3,1])
    ax.set_xlabel("momentum along the k-path")
    ax.set_ylabel("energy [$t$]")

style.save(draw, "operators_spin")
