import style
from pyqula import geometry
import numpy as np
g = geometry.honeycomb_lattice().get_supercell(80) # 12800 sites
g.dimensionality = 0 # a finite island with zigzag edges
h = g.get_hamiltonian(is_sparse=True,has_spin=False) # keep it sparse
(x,y,es,ldos) = h.get_multildos(energies=np.linspace(-1.,1.,41), # energies
                    mode="KPM", # Chebyshev expansion of every site
                    delta=0.1, # energy resolution, half width of the peak of a level
                    write=False) # return the maps without writing them

# the sites at the edge have two neighbors instead of three
edge = h.intra.getnnz(axis=1) < 3 # number of hoppings from each site
print("edge sites:",edge.sum(),"of",len(x))
for i in [20,30]: # E = 0 and E = 0.5
    m = ldos[i]
    print("E =",es[i],"edge weight:",m[edge].sum()/m.sum(),
          "bulk median:",np.median(m[~edge]),"edge median:",np.median(m[edge]),
          "max:",m.max())

from matplotlib.colors import LogNorm, Normalize
def draw(axs, c):
    for (ax,i) in zip(axs, [20, 30]): # E = 0 and E = 0.5
        m = ldos[i]
        o = np.argsort(m) # the brightest sites are drawn last, on top
        vmax = np.percentile(m, 99.5) # the brightest edge sites saturate
        if i==20: norm = LogNorm(vmin=vmax/1000, vmax=vmax) # three decades at E=0
        else: norm = Normalize(vmin=0, vmax=vmax)
        s = ax.scatter(x[o], y[o], c=m[o], s=4, lw=0, cmap=c.cmap, norm=norm)
        style.colorbar(s, ax, "local density of states")
        ax.set_title("$E=%.1f\\,t$" % es[i])
        ax.set_aspect("equal")
        ax.set_axis_off()

style.save(draw, "kpm_ldos", npanels=2)
