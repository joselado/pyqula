import style
from pyqula import geometry
from pyqula import topology
import numpy as np
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
(es,cs,csi) = topology.chern_density(h,nk=10,es=np.linspace(-1.0,1.0,40))
(k,e) = h.get_bands() # bands, to mark the gap
gap = np.min(e[e>0]) # the gap is from -gap to gap

def draw(ax, c):
    ax[0].plot(es, cs, color=c.blue, marker="o", ms=4)
    ax[1].plot(es, csi, color=c.orange, marker="o", ms=4)
    ax[0].set_ylabel("Berry curvature density")
    ax[1].set_ylabel("cumulative integral")
    for axi in ax:
        axi.axvspan(-gap, gap, color=c.ink2, alpha=0.15, lw=0) # the bulk gap
        axi.axhline(0, color=c.ink, lw=0.6, alpha=0.4)
        axi.set_xlabel("energy [$t$]")
        axi.set_xlim(-1, 1)
        axi.set_xticks([-1,-0.5,0,0.5,1])

style.save(draw, "geometry_chern_density", npanels=2)
