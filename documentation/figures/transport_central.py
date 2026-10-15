import numpy as np
import style
from pyqula import geometry

g = geometry.chain()
gc = g.get_supercell(5)
gc.dimensionality = 0 # a finite, 5-site cluster (no periodicity)
hc = gc.get_hamiltonian() # the central region, can be any 0d Hamiltonian

h_normal = g.get_hamiltonian() # normal lead
h_sc = g.get_hamiltonian(); h_sc.add_swave(0.05) # superconducting lead

ht = hc.get_central_heterostructure(0,4,left=h_normal,right=h_sc)
G = ht.didv(energy=0.02) # Andreev conductance, via the BdG scattering-matrix formula
es = np.linspace(-0.15,0.15,121) # energies across the gap, for the figure
Gs = [ht.didv(energy=e) for e in es]

def draw(ax, c):
    ax.axvspan(-0.05, 0.05, color=c.ink2, alpha=0.12, lw=0)
    ax.plot(es, Gs, color=c.blue)
    ax.scatter([0.02], [G], color=c.orange, zorder=3, label="$E=0.02$")
    ax.set_xlabel("energy [$t$]")
    ax.set_ylabel("conductance [$e^2/h$]")
    ax.set_ylim(0, 4.5)
    ax.legend(loc="upper right")

style.save(draw, "transport_central")
