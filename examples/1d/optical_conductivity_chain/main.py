# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Optical (Kubo-Greenwood) conductivity of a one dimensional chain. A
# single band has no interband transitions, so the whole spectral weight
# sits in the Drude peak at zero frequency. Dimerizing the chain (a
# Peierls distortion, hopping 1+d and 1-d on alternating bonds) opens a
# gap at half filling and moves that weight out of the Drude peak and
# into an interband absorption edge at 2*(t1-t2). Either way the total
# weight is fixed by the f-sum rule: the integral of Re sigma_xx over all
# frequencies is pi times the diamagnetic weight

import numpy as np
from pyqula import geometry
from pyqula import conductivity
from pyqula.multihopping import MultiHopping

energies = np.linspace(-4.0,4.0,400) # frequencies (Re sigma_xx is even)
wide = np.linspace(-60.,60.,6001) # wide window, for the f-sum rule
                                  # (the Lorentzian tails decay slowly)


def dimerized_chain(d):
    """Chain with alternating hoppings 1+d and 1-d, in a two site cell"""
    g = geometry.chain().get_supercell(2)
    h = g.get_hamiltonian(has_spin=False)
    dd = h.get_multihopping().get_dict() # real space hoppings
    for key in dd: # intracell bond gets 1+d, intercell bond 1-d
        dd[key] = dd[key]*((1.+d) if tuple(key)==(0,0,0) else (1.-d))
    h.set_multihopping(MultiHopping(dd))
    return h


out = []
for d in [0.0,0.4]: # uniform chain and dimerized chain
    h = dimerized_chain(d)
    (ws,sigma) = conductivity.optical_conductivity(h,energies=energies,
                                      nk=400,T=0.05,delta=0.1)
    D = conductivity.drude_weight(h,nk=400,T=0.05)[0,0]
    W = conductivity.sum_rule_weight(h,nk=400,T=0.05)[0,0]
    sw = conductivity.optical_conductivity(h,energies=wide,nk=400,
                                      T=0.05,delta=0.1)[1]
    print("Dimerization",d,"  gap",round(h.get_gap(),4))
    print("   Drude weight            ",round(D,4))
    print("   f-sum rule weight       ",round(W,4))
    print("   integral Re sigma_xx/pi ",
          round(np.trapezoid(sw[:,0,0].real,wide)/np.pi,4))
    out.append((ws,sigma,d))

import matplotlib.pyplot as plt

for (ws,sigma,d) in out:
    plt.plot(ws,sigma[:,0,0].real,label="dimerization = "+str(d))
plt.xlabel("Frequency") ; plt.ylabel("Re $\\sigma_{xx}$ [$e^2 a/\\hbar$]")
plt.legend()
plt.show()
