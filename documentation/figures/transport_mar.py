import style
from pyqula import geometry
from pyqula import heterostructures
import numpy as np
g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create the Hamiltonian
h1 = h.copy() ; h1.add_swave(0.1) # left superconducting lead
h2 = h.copy() ; h2.add_swave(0.1) # right superconducting lead
HT = heterostructures.build(h1,h2) # junction between the two superconductors
HT.set_coupling(0.5) # set the normal transparency of the weak link
vs = np.linspace(0.02,1.5,40)*0.1 # bias voltages
Is = HT.get_iv_curve(vs) # MAR/AC-Josephson dc current
for (v,I) in zip(vs,Is): print(v,I)

def draw(ax, c):
    for n in [2,3,4]: # MAR features at eV = 2 Delta/n
        ax.axvline(2/n, color=c.ink2, lw=0.8, ls="--")
        ax.text(2/n, 0.235-0.012*n, f" $2\\Delta/{n}$", color=c.ink2, ha="left")
    ax.plot(vs/0.1, Is, "o-", color=c.blue)
    ax.set_ylim(0, 0.23)
    ax.set_xlabel(r"bias voltage $eV$ [$\Delta$]")
    ax.set_ylabel(r"dc current $I_{dc}$ [$e t/h$]")

style.save(draw, "transport_mar")
