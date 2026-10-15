import numpy as np
import style
from pyqula import specialhamiltonian

h = specialhamiltonian.square_altermagnet(am=1.)

# largest spin splitting in each energy bin, over the whole BZ
(es, ds) = h.get_spin_splitting_vs_energy(nk=800, nbins=400)
print("largest spin splitting anywhere in the BZ:", ds.max())

# the smooth counterpart, on the same axes
(xs, ys) = h.get_spin_splitting_density(nk=100, delta=1e-1, energies=es)

nodes = np.array([[0,0,0],[.5,0,0],[.5,.5,0],[0,0,0],[0,.5,0]]) # Gamma X M Gamma Y
n = [int(200*np.linalg.norm(b-a)) for (a,b) in zip(nodes,nodes[1:])] # points per segment
kpath = np.concatenate([np.linspace(a,b,m,endpoint=False) for (a,b,m) in zip(nodes,nodes[1:],n)])
(k,e,sz) = h.get_bands(operator="sz", kpath=kpath) # bands, and the spin of each state
ticks = np.cumsum([0]+n) # index of each high-symmetry point

def draw(axs, c):
    ax = axs[0]
    sc = ax.scatter(k, e, c=sz, s=6, cmap=c.diverging, vmin=-1, vmax=1)
    style.colorbar(sc, ax, r"spin $\langle S_z\rangle$")
    ax.set_xticks(ticks, [r"$\Gamma$", "X", "M", r"$\Gamma$", "Y"])
    ax.set_xlim(0, ticks[-1])
    ax.set_xlabel("momentum along the k-path")
    ax.set_ylabel("energy [$t$]")
    ax = axs[1]
    ax.plot(es, ds, color=c.blue, label="largest splitting")
    ax.plot(xs, ys, color=c.orange, label="splitting density")
    ax.set_xlabel("mean energy $\\bar E$ [$t$]")
    ax.set_ylabel("spin splitting $\\Delta$ [$t$]")
    ax.set_ylim(0, 5.5)
    ax.legend(loc="upper left")

style.save(draw, "altermagnets_splitting", npanels=2)
