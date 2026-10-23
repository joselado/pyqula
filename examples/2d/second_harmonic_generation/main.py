# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Second-harmonic generation (SHG) of a two-band model of hBN, the
# honeycomb lattice with onsite energies +-3.9 eV and hopping t = 2.33 eV.
# Light at frequency w gives a current at 2w,
#   J_a(2w) = sigma_abc E_b(w) E_c(w),
# and the susceptibility chi_abc = i sigma_abc/(2 w eps0) is what an SHG
# experiment measures. The point group is D3h, and with x along the
# armchair direction (pyqula's honeycomb lattice) the only nonzero
# components are chi_xxx = -chi_xyy = -chi_yxy = -chi_yyx.
# Resonances appear when 2w or w matches a transition between bands.

import numpy as np
from pyqula import geometry

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True) # spinful, both spins are counted
h.add_sublattice_imbalance(3.9/2.33) # staggered onsite energy, in units of t

comps = ["xxx","xyy","yyy"]
(ws,chi) = h.get_shg_susceptibility(energies=np.linspace(0.,4.,200),
                                    nk=100,T=0.,delta=0.02,components=comps,
                                    length_unit=1.45,energy_unit=2.33)
print("static chi_xxx in nm^2/V ",chi["xxx"][0].real)
print("chi_xyy/chi_xxx          ",(chi["xyy"][0]/chi["xxx"][0]).real) # -1
print("max |chi_yyy| in nm^2/V  ",np.max(np.abs(chi["yyy"]))) # zero by symmetry

import matplotlib.pyplot as plt

gap = h.get_gap()
plt.plot(ws,chi["xxx"].real,c="blue",label="Re $\\chi_{xxx}$")
plt.plot(ws,chi["xxx"].imag,c="red",label="Im $\\chi_{xxx}$")
plt.axvline(gap/2.,c="black",ls="--") # 2w resonance with the gap
plt.axvline(gap,c="black",ls=":") # w resonance with the gap
plt.xlabel("Frequency [t]") ; plt.ylabel("sheet $\\chi^{(2)}$ [nm$^2$/V]")
plt.legend()
plt.tight_layout()
plt.show()
