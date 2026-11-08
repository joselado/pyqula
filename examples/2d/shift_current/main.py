# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Bulk photovoltaic effect of a gapped honeycomb lattice. Light of
# frequency w, E(t) = Re E0 exp(-i w t), gives a dc current at second order,
#   J_a = (1/2) sigma_abc(0;w,-w) E0_b E0_c^*,
# made of two parts. The shift current stays finite when the broadening
# delta goes to zero. The injection current grows in time as
# dJ_a/dt = eta_abc E0_b E0_c^* until the relaxation stops it.
# The gapped honeycomb lattice has time reversal, so the linear injection
# is zero and the shift current is the whole response; its point group
# D3h allows only xxx = -xyy = -yxy = -yyx. The Haldane term breaks time
# reversal and gives a linear injection.

import numpy as np
from pyqula import geometry

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False) # spinless, one spin species
h.add_sublattice_imbalance(0.5) # staggered onsite energy, gap 1.0

ws = np.linspace(0.,4.,201) # frequencies
(ws,s) = h.get_shift_current(energies=ws,nk=100,T=0.,delta=0.05,
                             components=["xxx","xyy","yyy"])
i = np.argmax(np.abs(s["xxx"]))
print("largest |sigma_xxx| of the shift current",abs(s["xxx"][i]),"at w =",ws[i])
print("sigma_xyy/sigma_xxx",(s["xyy"][i]/s["xxx"][i]).real) # -1
print("largest |sigma_yyy|",np.max(np.abs(s["yyy"]))) # zero by symmetry

# injection rate with broken time reversal
h2 = g.get_hamiltonian(has_spin=False)
h2.add_haldane(0.2) # second-neighbor Haldane term
h2.add_sublattice_imbalance(0.3) # gap 1.2
(ws,eta) = h2.get_injection_current(energies=ws,nk=100,T=0.,delta=0.05,
                                    components=["yyy","yxx","xxx"])
j = np.argmax(np.abs(eta["yyy"]))
print("largest |eta_yyy|",abs(eta["yyy"][j]),"at w =",ws[j])
print("eta_yxx/eta_yyy",(eta["yxx"][j]/eta["yyy"][j]).real) # -1
print("largest |eta_xxx|",np.max(np.abs(eta["xxx"]))) # zero by symmetry

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(ws,s["xxx"].real,c="blue",label="shift, $\\sigma_{xxx}$")
plt.axvline(h.get_gap(),c="black",ls="--") # absorption edge
plt.xlabel("Frequency [t]") ; plt.ylabel("$\\sigma^{shift}$")
plt.legend()
plt.subplot(1,2,2)
plt.plot(ws,eta["yyy"].real,c="red",label="injection, $\\eta_{yyy}$")
plt.axvline(h2.get_gap(),c="black",ls="--") # absorption edge
plt.xlabel("Frequency [t]") ; plt.ylabel("$\\eta$")
plt.legend()
plt.tight_layout()
plt.show()
