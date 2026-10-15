import style
from pyqula import geometry
from pyqula import heterostructures
import numpy as np
g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create the Hamiltonian
es = np.linspace(-.5,.5,50) # set of energies for dIdV
Gs = dict()
for name,m2 in [("parallel",[0.,0.,0.5]),("antiparallel",[0.,0.,-0.5])]:
    h1 = h.copy() ; h1.add_exchange([0.,0.,0.5]) # first lead, fixed magnetization
    h2 = h.copy() ; h2.add_exchange(m2) # second lead, parallel or antiparallel
    HT = heterostructures.create_leads_and_central(h1,h2,h1) # create the junction
    Gs[name] = [HT.didv(energy=e) for e in es] # calculate conductance
GP,GAP = np.array(Gs["parallel"]),np.array(Gs["antiparallel"])

def draw(ax, c):
    ax[0].plot(es, GP, color=c.blue, label="parallel")
    ax[0].plot(es, GAP, color=c.orange, label="antiparallel")
    ax[0].set_xlabel("energy [$t$]")
    ax[0].set_ylabel("conductance [$e^2/h$]")
    ax[0].set_ylim(1.8, 2.05)
    ax[0].legend(loc="center")
    ax[1].plot(es, 100*(GP-GAP)/GAP, color=c.green)
    ax[1].set_xlabel("energy [$t$]")
    ax[1].set_ylabel("magnetoresistance [%]")

style.save(draw, "transport_magnetoresistance", npanels=2)
