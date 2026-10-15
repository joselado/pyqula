import numpy as np
import style
from pyqula import geometry
np.random.seed(0) # fix the random guess, so that the figure is reproducible
g = geometry.square_lattice() # geometry of a square lattice
g = g.get_supercell([2,2]) # generate a 2x2 supercell
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_zeeman([0.,0.,0.1]) # add out-of-plane Zeeman field
h = h.get_mean_field_hamiltonian(U=2.0,filling=0.5,
                                  mf="random") # perform SCF
(k,e,c) = h.get_bands(operator="sz") # calculate band structure
m = h.get_magnetization() # get the magnetization
n = np.array([m[0,0], m[0,1], 0.])/np.linalg.norm(m[0,:2]) # in-plane Neel axis
mn, mz = m @ n, m[:,2] # each moment, along the Neel axis and along z

def draw(axs, col):
    sc = axs[0].scatter(k, e, c=c, s=3, cmap=col.diverging, vmin=-1, vmax=1)
    style.colorbar(sc, axs[0], r"spin $\langle S_z\rangle$")
    nk = k.max() # the path goes Gamma-M-Gamma
    axs[0].set_xticks([0, nk/2, nk], [r"$\Gamma$", "M", r"$\Gamma$"])
    axs[0].set_xlim(0, nk)
    axs[0].set_xlabel("momentum")
    axs[0].set_ylabel("energy [$t$]")
    axs[1].quiver(0*mn, 0*mz, mn, mz, color=col.blue, angles="xy",
                  scale_units="xy", scale=1, width=0.012)
    axs[1].quiver(0, 0, 0, 0.3, color=col.orange, angles="xy",
                  scale_units="xy", scale=1, width=0.008)
    axs[1].text(0.02, 0.25, "Zeeman field", color=col.orange)
    axs[1].set_xlim(-0.45, 0.45)
    axs[1].set_ylim(-0.35, 0.35)
    axs[1].set_aspect("equal")
    axs[1].set_xlabel("moment along the Neel axis")
    axs[1].set_ylabel("moment along $z$")

style.save(draw, "meanfield_noncollinear", npanels=2)
print(m)
