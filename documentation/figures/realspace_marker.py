import style
from pyqula import islands
from pyqula import topology
import numpy as np
from pyqula.topologytk.realspace import site_areas
g = islands.get_geometry(name="honeycomb",n=16,nedges=3) # a honeycomb island, larger than on the page
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
(r,c) = topology.real_space_chern(h) # local marker, per site
r = np.array(r)
bulk = np.argsort(np.linalg.norm(r-r.mean(axis=0),axis=1))[:len(r)//4] # innermost sites
C = np.mean(c[bulk]) # bulk plateau value approximates the total Chern number
d = np.linalg.norm(r-r.mean(axis=0),axis=1) # distance of each site to the center
A = site_areas(g) # area of each site
order = np.argsort(d) # sites from the center outwards
total = np.cumsum((c*A)[order]) # marker times area, summed up to a distance

def draw(axs, cl):
    ax = axs[0]
    ax.scatter(d, c, s=6, color=cl.blue)
    ax.scatter(d[bulk], c[bulk], s=6, color=cl.orange, label="innermost quarter")
    ax.axhline(C, color=cl.ink, lw=1, ls="--", label=f"mean = {C:.2f}")
    ax.axhline(0, color=cl.ink, lw=0.6, alpha=0.4)
    ax.set_xlabel("distance to the center [$d$]")
    ax.set_ylabel(r"local Chern marker $C(\mathbf{r})$")
    ax.legend(loc="lower left")
    axs[1].plot(d[order], total, color=cl.blue)
    axs[1].plot(d[order], 2*np.cumsum(A[order]), color=cl.ink2, lw=1, ls="--",
                label="$C=2$ times the area") # what a bulk of marker 2 would give
    axs[1].set_ylim(bottom=-0.05*total.max(), top=1.15*total.max())
    axs[1].legend(loc="upper left")
    axs[1].axhline(0, color=cl.ink, lw=0.6, alpha=0.4)
    axs[1].set_xlabel("radius $R$ [$d$]")
    axs[1].set_ylabel(r"$\sum_{|\mathbf{r}|<R}\, C(\mathbf{r})A_\mathbf{r}$")

style.save(draw, "realspace_marker", npanels=2)
