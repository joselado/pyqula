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
es = np.linspace(0.15,0.25,40)*0.1 # bias energies
Gs = HT.didv(energies=es, nmax_max=40) # dI/dV curve
for (e,G) in zip(es,Gs): print(e,G)

def draw(ax, c):
    for n in range(8,14): # MAR features at eV = 2 Delta/n
        ax.axvline(2/n, color=c.ink2, lw=0.8, ls="--")
        ax.text(2/n, 0.29, f"{n}", color=c.ink2, ha="center")
    ax.set_title(r"dashed lines: $eV=2\Delta/n$, with $n$ on top", color=c.ink2)
    ax.plot(es/0.1, Gs, "o-", color=c.blue, ms=4)
    ax.set_ylim(0, 0.31)
    ax.set_xlabel(r"bias voltage $eV$ [$\Delta$]")
    ax.set_ylabel(r"conductance $dI_{dc}/dV$ [$e^2/h$]")

style.save(draw, "transport_finitev")
