import style
from pyqula import geometry
from pyqula import topology
import numpy as np
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
# the snippet of the page, converged: the energies start below the bottom of
# the band, with a step of delta/2, the k-mesh contains the K points (nk is a
# multiple of 3), and the finite-difference step dk is delta/25, the default
delta = 0.05
es = np.arange(-3.5,3.5+1e-9,delta/2)
(es,cs,csi,cex) = topology.chern_density(h,nk=48,es=es,delta=delta,
                                         exact_at=[0.]) # without broadening at E=0
print("plateau at zero energy:",np.interp(0.,es,csi),"exact:",cex[0])
(k,e) = h.get_bands() # bands, to mark the gap
gap = np.min(e[e>0]) # the gap is from -gap to gap

def draw(ax, c):
    ax[0].plot(es, cs, color=c.blue)
    ax[1].plot(es, csi, color=c.orange)
    ax[1].axhline(2, color=c.ink2, lw=1.0, ls="--") # the Chern number
    ax[1].plot([0.], cex, "o", color=c.blue, ms=7, label="exact_at=[0.]")
    ax[1].legend(loc="upper right")
    ax[0].set_ylabel("Berry curvature density")
    ax[1].set_ylabel("cumulative integral")
    for axi in ax:
        axi.axvspan(-gap, gap, color=c.ink2, alpha=0.15, lw=0) # the bulk gap
        axi.axhline(0, color=c.ink, lw=0.6, alpha=0.4)
        axi.set_xlabel("energy [$t$]")
        axi.set_xlim(-3.5, 3.5)

style.save(draw, "geometry_chern_density", npanels=2)
