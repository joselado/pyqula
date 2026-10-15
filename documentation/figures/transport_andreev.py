import style
from pyqula import geometry
from pyqula import heterostructures
import numpy as np
g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create the Hamiltonian
h1 = h.copy() # first lead
h2 = h.copy() # second lead
h2.add_swave(.01) # the second lead is superconducting
es = np.linspace(-.03,.03,100) # set of energies for dIdV
Ts = np.linspace(1e-3,1.0,16) # 16 transparencies instead of 6, for the map
Gmap = [] # conductance for each transparency
for T in Ts: # loop over transparencies
    HT = heterostructures.build(h1,h2) # create the junction
    HT.set_coupling(T) # set the coupling between the leads
    Gs = [HT.didv(energy=e) for e in es] # calculate conductance
    Gmap.append(np.array(Gs)/Gs[0]) # in units of the conductance outside the gap
Gmap = np.array(Gmap)

def draw(ax, c):
    im = ax[0].pcolormesh(es/0.01, Ts, Gmap, cmap=c.cmap, vmin=0, vmax=2.5,
                          shading="nearest")
    style.colorbar(im, ax[0], r"$G/G_N$")
    ax[0].set_xlabel(r"energy [$\Delta$]")
    ax[0].set_ylabel("transparency $T$")
    for i,col in zip([0,5,15], [c.blue, c.orange, c.green]):
        ax[1].plot(es/0.01, Gmap[i], color=col, label=f"$T={Ts[i]:.3g}$")
    ax[1].set_ylim(0, 3)
    ax[1].set_xlabel(r"energy [$\Delta$]")
    ax[1].set_ylabel(r"conductance $G/G_N$")
    ax[1].legend(loc="upper left")

style.save(draw, "transport_andreev", npanels=2)
