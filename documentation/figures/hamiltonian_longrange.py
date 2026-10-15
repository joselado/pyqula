import numpy as np
import style
from pyqula import geometry

g = geometry.chain() # geometry of the 1D chain
h1 = g.get_hamiltonian() # first-neighbor hopping only
h = g.get_hamiltonian(tij=[1.0,0.2,0.3]) # Hamiltonian with t1,t2,t3
(k,e1) = h1.get_bands()
(k,e) = h.get_bands() # compute band structure
k = 2*np.pi*k/k.max() # the k-path goes from 0 to 2 pi

def draw(ax, c):
    ax.scatter(k, e1, s=8, color=c.ink2, label=r"$t_2=t_3=0$")
    ax.scatter(k, e, s=8, color=c.blue, label=r"$t_2=0.2,\ t_3=0.3$")
    ax.axhline(0, color=c.ink, lw=0.6, alpha=0.4)
    style.kpath(ax, k, ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel("momentum $k$")
    ax.set_ylabel("energy [$t$]")
    ax.legend(loc="upper center")

style.save(draw, "hamiltonian_longrange")
