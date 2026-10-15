import style
from pyqula import geometry
import numpy as np
np.random.seed(1) # the same random vectors in every run
g = geometry.chain()
g = g.get_supercell(100000) # a big supercell
g.dimensionality = 0
h = g.get_hamiltonian(is_sparse=True,has_spin=False)
(x,y) = h.get_dos(mode="KPM",
            energies=np.linspace(-3.0,3.0,601), # energies
            delta=1e-2, # energy resolution, half width of the peak of a level
            ntries=40 # number of random vectors for the stochastic trace
            )
N = len(g.r) # number of sites
x0 = np.linspace(-1.999,1.999,400)
y0 = N/(np.pi*np.sqrt(4-x0**2)) # density of states of the infinite chain

x = np.array(x) ; inside = np.abs(x)<1.5 # far from the band edges
exact = 1/(np.pi*np.sqrt(4-x[inside]**2))
dev = np.abs(y[inside]/N/exact-1) # relative deviation from the infinite chain
print("deviation for |E|<1.5: mean",dev.mean(),"max",dev.max())
print("peak at the band edge:",y.max()/N,"at E =",x[np.argmax(y)])

def draw(ax, c):
    ax.plot(x, y/N, color=c.blue, label="KPM, $10^5$ sites")
    ax.plot(x0, y0/N, "--", color=c.ink, lw=1.2, label=r"$1/(\pi\sqrt{4t^2-E^2})$")
    ax.set_xlabel("energy [$t$]")
    ax.set_ylabel("density of states per site")
    ax.set_ylim(0, 1.6)
    ax.legend(loc="upper center")

style.save(draw, "kpm_dos")
