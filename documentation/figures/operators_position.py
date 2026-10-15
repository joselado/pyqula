import style
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # ribbon periodic along x
h = g.get_hamiltonian()  # get the Hamiltonian
h.add_haldane(0.1) # Haldane coupling, gives chiral states at the edges
y = h.get_operator("yposition") # y component
(k,e,yk) = h.get_bands(operator=y) # bands, and <y> of each state
W = g.y.max() # half width of the ribbon

def draw(ax, c):
    sc = ax.scatter(k, e, c=yk, s=4, cmap=c.diverging, vmin=-W, vmax=W)
    style.colorbar(sc, ax, r"position across the ribbon $\langle y\rangle$")
    style.kpath(ax, k, ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel("momentum along the ribbon")
    ax.set_ylabel("energy [$t$]")
    ax.set_ylim(-1.5, 1.5)

style.save(draw, "operators_position")
