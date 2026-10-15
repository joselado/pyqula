import style
from pyqula import geometry

N = 20 # number of unit cells as the width
g = geometry.square_ribbon(N) # ribbon
h = g.get_hamiltonian() # generate the Hamiltonian
B = 0.02 # magnetic field in quantum flux unit
h.add_orbital_magnetic_field(B) # add an out-of plane magnetic field
(k,e,y) = h.get_bands(operator="yposition") # bands, and position of each state

def draw(ax, c):
    sc = ax.scatter(k, e, c=y, s=5, cmap=c.diverging, vmin=-N/2, vmax=N/2)
    style.colorbar(sc, ax, r"position across the ribbon $\langle y\rangle$")
    style.kpath(ax, k, ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel("momentum along the ribbon")
    ax.set_ylabel("energy [$t$]")
    ax.set_ylim(-4, -2)

style.save(draw, "hamiltonian_landau")
