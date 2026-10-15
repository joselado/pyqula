import numpy as np
import style
from pyqula import geometry

g = geometry.chain() # geometry of the 1D chain
h = g.get_hamiltonian() # generate the Hamiltonian
(k,e) = h.get_bands() # compute band structure
k = 2*np.pi*k/k.max() # the k-path goes from 0 to 2 pi

def draw(ax, c):
    ax.scatter(k, e, s=8, color=c.blue, label="pyqula")
    ax.plot(k, 2*np.cos(k), "--", color=c.ink, lw=1.2, label=r"$2t\cos k$")
    style.kpath(ax, k, ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel("momentum $k$")
    ax.set_ylabel("energy [$t$]")
    ax.legend(loc="upper center")

style.save(draw, "hamiltonian_chain")
