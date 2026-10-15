import numpy as np
import style
from pyqula import geometry
from pyqula import latticeising
np.random.seed(1) # fixed seed, so that every run gives the same figure

g = geometry.square_lattice() # bipartite, so ferromagnetic order is not frustrated
g = g.get_supercell(12) # periodic, so no site sits at an edge

li = latticeising.LatticeIsing(g,m=0.0) # random +-1 spins, zero net magnetization
li.add_interaction(Jij=[1.]) # first-neighbor ferromagnetic coupling
temps = [6.,3.,1.,0.3,0.1,0.03] # the figure starts above Tc=4.54
es,ms = li.anneal(temps=temps,ntries=1e4,checkpoint_at=10000) # simulated annealing
s_hot = li.checkpoints[10000] # spins at the end of the stage at T=6

def spins(ax, s, title, c):
    sc = ax.scatter(g.r[:,0], g.r[:,1], c=s, s=120, marker="s",
                    cmap=c.diverging, vmin=-1, vmax=1)
    ax.set_aspect("equal") ; ax.set_axis_off()
    ax.set_title(title)
    return sc

def draw(ax, c):
    n = len(ms)//len(temps) # Monte Carlo steps per temperature
    ax[0].plot(np.arange(len(ms))/1e3, ms/li.nsites, color=c.blue)
    for i,T in enumerate(temps):
        ax[0].axvline(i*n/1e3, color=c.ink2, lw=0.6, alpha=0.5)
        ax[0].text((i+0.5)*n/1e3, 1.2, f"{T:g}", ha="center", color=c.ink2)
    ax[0].set_title("temperature [$J$]", color=c.ink2)
    ax[0].set_ylim(-1.1, 1.35)
    ax[0].set_xlabel("Monte Carlo step [$10^3$]")
    ax[0].set_ylabel(r"magnetization per site")
    spins(ax[1], s_hot, "$T=6$", c)
    sc = spins(ax[2], li.s, "$T=0.03$", c)
    style.colorbar(sc, ax[2], "spin $s_i$")

style.save(draw, "ising_ferro", npanels=3)
