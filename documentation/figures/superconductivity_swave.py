import style
from pyqula import geometry
g = geometry.triangular_lattice() # geometry of the 2D model
h = g.get_hamiltonian() # generate the Hamiltonian
h.add_swave(0.2) # add s-wave superconductivity
(k,e,c) = h.get_bands(operator="electron",nk=3000) # bands, and electron weight of each state

def draw(axs, col):
    n = k.max() # the path goes Gamma-K-M-K'-Gamma, K at one third
    for ax in axs:
        sc = ax.scatter(k, e, c=2*c-1, s=3, cmap=col.diverging, vmin=-1, vmax=1)
        ax.set_xticks([0, n/3, n/2, 2*n/3, n], [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"])
        ax.set_xlim(0, n)
        ax.set_xlabel("momentum")
    axs[0].set_ylabel("energy [$t$]")
    kf = k[k < n/3][abs(e[k < n/3]).argmin()] # where the bands cross zero
    axs[1].set_xlim(kf - n/40, kf + n/40) # zoom on the gap
    axs[1].set_ylim(-0.8, 0.8)
    axs[1].set_xticks([])
    axs[1].set_xlabel("momentum, close to the crossing")
    for y in [-0.2, 0.2]: axs[1].axhline(y, color=col.ink2, lw=0.8, ls="--")
    style.colorbar(sc, axs[1], "hole (-1) to electron (+1)")

style.save(draw, "superconductivity_swave", npanels=2)
