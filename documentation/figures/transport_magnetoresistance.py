import style
from pyqula import geometry
from pyqula import heterostructures
import numpy as np
g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create the Hamiltonian
h.add_onsite(1.0) # the band is now [-1,3], and E=0 is close to its bottom
thetas = np.linspace(0.,np.pi,61) # angles between the two magnetizations
def conductance(J,theta): # conductance at E=0 for exchange J and angle theta
    h1 = h.copy() ; h1.add_exchange([0.,0.,J]) # first lead, magnetization along z
    h2 = h.copy() ; h2.add_exchange([J*np.sin(theta),0.,J*np.cos(theta)]) # rotated in the xz plane
    HT = heterostructures.create_leads_and_central(h1,h2,h1) # create the junction
    return HT.didv(energy=0.0) # calculate conductance
Js = [0.3,0.6,0.9,1.5] # exchange strengths, the last one is a half metal
Gs = {J:np.array([conductance(J,t) for t in thetas]) for J in Js}
for J in Js: print("J =",J,"G_P =",Gs[J][0],"G_AP =",Gs[J][-1])

Jmap = np.linspace(0.,2.,81) # exchange strengths for the map
Gmap = np.array([[conductance(J,t) for t in thetas] for J in Jmap])

def draw(ax, c):
    for J,col in zip(Js, [c.blue, c.orange, c.green, c.magenta]):
        ax[0].plot(thetas/np.pi, Gs[J], color=col, label=f"$J={J:g}$")
    ax[0].plot(thetas/np.pi, np.cos(thetas/2)**2, ":", color=c.ink, lw=1.2,
               label=r"$\cos^2(\theta/2)$")
    ax[0].set_xlabel(r"angle $\theta$ [$\pi$]")
    ax[0].set_ylabel("conductance [$e^2/h$]")
    ax[0].set_ylim(0, 2.1)
    ax[0].legend(loc="lower left")
    im = ax[1].pcolormesh(thetas/np.pi, Jmap, Gmap, cmap=c.cmap, vmin=0, vmax=2,
                          shading="nearest")
    style.colorbar(im, ax[1], "conductance [$e^2/h$]")
    ax[1].axhline(1.0, color=c.ink2, lw=0.8, ls="--")
    ax[1].set_xlabel(r"angle $\theta$ [$\pi$]")
    ax[1].set_ylabel("exchange $J$ [$t$]")

style.save(draw, "transport_magnetoresistance", npanels=2)
