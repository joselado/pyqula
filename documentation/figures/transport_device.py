import numpy as np
import style
from pyqula import geometry
from pyqula import multiterminal
np.random.seed(1) # fixed seed, so that every run gives the same figure

def lead_cell(x): # a one-site chain unit cell sitting at x
    g = geometry.chain() ; g.r = np.array([[x,0.,0.]]) ; g.r2xyz()
    return g

gc = geometry.chain().supercell(4) # the scattering region
xs = gc.r[:,0]
d = multiterminal.Device()
d.biterminal(left_g=lead_cell(min(xs)-1.),right_g=lead_cell(max(xs)+1.),
             central_g=gc,disorder=0.0)
print("ballistic transmission:",d.transmission(energy=0.0)[0])

es = np.linspace(-2.5,2.5,51) # energies for the figure
Ws = np.linspace(0.,4.,13) # disorder strengths
Tmap = np.zeros((len(Ws),len(es))) # transmission, averaged over disorder
for i,W in enumerate(Ws):
    for n in range(30): # 30 random configurations
        d = multiterminal.Device()
        d.biterminal(left_g=lead_cell(min(xs)-1.),right_g=lead_cell(max(xs)+1.),
                     central_g=gc,disorder=W)
        Tmap[i] += [d.transmission(energy=e)[0]/30 for e in es]

def draw(ax, c):
    ax[0].plot(es, Tmap[0], color=c.blue, label="clean")
    ax[0].plot(es, Tmap[-1], color=c.orange, label=f"disorder {Ws[-1]:g}")
    ax[0].set_xlabel("energy [$t$]")
    ax[0].set_ylabel("transmission")
    ax[0].set_ylim(0, 1.1)
    ax[0].legend(loc="lower center")
    im = ax[1].pcolormesh(es, Ws, Tmap, cmap=c.cmap, vmin=0, vmax=1, shading="nearest")
    style.colorbar(im, ax[1], "transmission")
    ax[1].set_xlabel("energy [$t$]")
    ax[1].set_ylabel("disorder strength [$t$]")

style.save(draw, "transport_device", npanels=2)
