import numpy as np
import style
from pyqula import geometry
np.random.seed(0) # fix the random guess, so that the figure is reproducible
g = geometry.square_lattice() # geometry of a square lattice
g = g.get_supercell([2,2]) # generate a 2x2 supercell
h = g.get_hamiltonian() # create hamiltonian of the system
h = h.get_mean_field_hamiltonian(U=2.0,filling=0.5,
                            mf="random") # perform SCF
(k,e) = h.get_bands() # calculate band structure
m = h.get_magnetization() # get the magnetization
ms = m @ m[0]/np.linalg.norm(m[0]) # moment of each site along the one of site 0
r = np.array([ri + i*g.a1 + j*g.a2 for i in range(3) for j in range(3) for ri in g.r])

def draw(axs, c):
    axs[0].scatter(k, e, s=3, color=c.blue)
    axs[0].axhline(0, color=c.ink, lw=0.6, alpha=0.4)
    n = k.max() # the path goes Gamma-M-Gamma
    axs[0].set_xticks([0, n/2, n], [r"$\Gamma$", "M", r"$\Gamma$"])
    axs[0].set_xlim(0, n)
    axs[0].set_xlabel("momentum")
    axs[0].set_ylabel("energy [$t$]")
    sc = axs[1].scatter(r[:,0], r[:,1], c=np.tile(ms, 9), s=300, marker="s",
                        cmap=c.diverging, vmin=-0.5, vmax=0.5)
    axs[1].set_aspect("equal")
    axs[1].axis("off")
    style.colorbar(sc, axs[1], "moment along the Neel axis")

style.save(draw, "meanfield_collinear", npanels=2)
print("moments", ms)
print("gap", e[e>0].min()-e[e<0].max())
