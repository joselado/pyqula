import style
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
vall = h.get_operator("valley") # valley operator
(k,e,v) = h.get_bands(operator=vall) # bands, and valley of each state

def draw(ax, c):
    sc = ax.scatter(k, e, c=v, s=4, cmap=c.diverging, vmin=-1, vmax=1)
    style.colorbar(sc, ax, r"valley $\langle \tau_z\rangle$")
    style.kpath(ax, k, ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel("momentum along the ribbon")
    ax.set_ylabel("energy [$t$]")
    ax.set_ylim(-1.5, 1.5)

style.save(draw, "operators_valley")
