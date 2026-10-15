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

es = np.linspace(-2.5,2.5,81) # energies for the figure
Ws = np.linspace(0.,4.,41) # disorder strengths, in steps of 0.1
nconf = 400 # random configurations for each disorder strength
Tmap = np.zeros((len(Ws),len(es))) # transmission, averaged over disorder
Terr = np.zeros((len(Ws),len(es))) # statistical error of the average
for i,W in enumerate(Ws):
    Ts = []
    for n in range(nconf):
        d = multiterminal.Device()
        d.biterminal(left_g=lead_cell(min(xs)-1.),right_g=lead_cell(max(xs)+1.),
                     central_g=gc,disorder=W)
        Ts.append([d.transmission(energy=e)[0] for e in es])
    Tmap[i] = np.mean(Ts,axis=0)
    Terr[i] = np.std(Ts,axis=0)/np.sqrt(nconf)
i0 = np.argmin(np.abs(es)) # E=0
for i in [0,10,20,30,40]: print("W =",Ws[i],"T(E=0) =",Tmap[i,i0],"+-",Terr[i,i0])
print("largest statistical error:",Terr.max())

def draw(ax, c):
    ax[0].plot(es, Tmap[0], color=c.blue, label="clean")
    for i,col in [(20,c.green),(40,c.orange)]:
        ax[0].plot(es, Tmap[i], color=col, label=f"disorder {Ws[i]:g}")
        ax[0].fill_between(es, Tmap[i]-Terr[i], Tmap[i]+Terr[i], color=col, alpha=0.3, lw=0)
    ax[0].set_xlabel("energy [$t$]")
    ax[0].set_ylabel("transmission")
    ax[0].set_ylim(0, 1.1)
    ax[0].legend(loc="lower center")
    im = ax[1].pcolormesh(es, Ws, Tmap, cmap=c.cmap, vmin=0, vmax=1, shading="nearest")
    style.colorbar(im, ax[1], "transmission")
    ax[1].set_xlabel("energy [$t$]")
    ax[1].set_ylabel("disorder strength [$t$]")

style.save(draw, "transport_device", npanels=2)
