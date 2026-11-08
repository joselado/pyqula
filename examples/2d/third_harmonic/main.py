# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Third-harmonic generation of doped graphene. Light at frequency w gives
# a current at 3w at third order in the field,
#   J_a(3w) = (1/4) Re[sigma_abcd(w,w,w) E0_b E0_c E0_d exp(-3 i w t)].
# The third order does not need broken inversion, so it is allowed in every
# crystal. Graphene has no second harmonic, but it has a third harmonic,
# and so does hBN. The point group D6h of graphene leaves one independent
# in-plane number,
#   sigma_xxxx = sigma_yyyy = sigma_xxyy + sigma_xyxy + sigma_xyyx,
# and the three last ones are equal since the fields are at the same
# frequency, so each one is sigma_xxxx/3.
# Every frequency carries the broadening delta, and the Fermi surface needs
# T > 0 and a mesh whose energy spacing is below T and delta.

import numpy as np
from pyqula import geometry

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False) # spinless, one spin species
h.shift_fermi(-0.5) # chemical potential at mu = 0.5

comps = ["xxxx","xxyy","xyxy","xyyx","yyyy"]
(ws,s) = h.get_nonlinear_conductivity(order=3,energies=np.linspace(0.1,3.,59),
                                      nk=100,T=0.05,delta=0.1,components=comps)
print("D6h, xxyy+xyxy+xyyx-xxxx:",np.max(np.abs(s["xxyy"]+s["xyxy"]+s["xyyx"]-s["xxxx"])))
print("xxyy/xxxx",(s["xxyy"][10]/s["xxxx"][10]).real) # 1/3

# the sheet susceptibility chi^(3) in nm^3/V^2, with bond length 1.42 A and t = 2.7 eV
(ws,chi) = h.get_nonlinear_susceptibility(order=3,energies=ws,nk=100,T=0.05,
                                          delta=0.1,components="xxxx",
                                          length_unit=1.42,energy_unit=2.7)

import matplotlib.pyplot as plt

plt.semilogy(ws,np.abs(s["xxxx"]),c="blue",label="$|\\sigma_{xxxx}(\\omega,\\omega,\\omega)|$")
plt.axvline(2*0.5/3.,c="black",ls="--") # 3w = 2 mu
plt.axvline(2*0.5,c="black",ls=":") # w = 2 mu
plt.xlabel("Frequency [t]") ; plt.ylabel("third-order conductivity")
plt.legend()
plt.tight_layout()
plt.show()
