# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Second-harmonic generation of MoS2 bilayers from the three-band model of
# Liu et al., with an interlayer hopping t_perp between the d_z2 orbitals
# of neighboring layers. In the 2H stacking the bilayer has inversion
# symmetry and the response is zero. In the R stacking it has none, and
# the two layers add. A potential step V between the two layers of the 2H
# bilayer breaks inversion, but with t_perp coupling only the d_z2
# orbitals, which do not couple the K valleys, the in-plane response it
# gives is small.
# x is the zigzag direction, so chi_yyy is the in-plane component.

import numpy as np
from pyqula import specialhamiltonian

tp = 0.1 # interlayer hopping in eV
ws = np.linspace(0.,1.5,76) # fundamental frequencies in eV
kw = dict(energies=ws,nk=60,T=0.,delta=0.03,components="yyy")

hm = specialhamiltonian.three_band_tmdc("MoS2") # the monolayer
(w,chim) = hm.get_shg_susceptibility(**kw)
h2H = specialhamiltonian.three_band_tmdc_multilayer(["MoS2","MoS2"],tp)
(w,chi2H) = h2H.get_shg_susceptibility(**kw)
hR = specialhamiltonian.three_band_tmdc_multilayer(["MoS2","MoS2"],tp,
                                                   stacking="R")
(w,chiR) = hR.get_shg_susceptibility(**kw)
hV = specialhamiltonian.three_band_tmdc_multilayer(["MoS2","MoS2"],tp,
                                                   onsite=[0.,0.1])
(w,chiV) = hV.get_shg_susceptibility(**kw)

print("static |chi_yyy| in nm^2/V")
print("monolayer          ",abs(chim[0]))
print("2H bilayer         ",abs(chi2H[0])) # zero, inversion symmetric
print("R bilayer          ",abs(chiR[0]))
print("2H bilayer, V=0.1  ",abs(chiV[0]))

import matplotlib.pyplot as plt
plt.plot(w,np.abs(chim),c="black",label="monolayer")
plt.plot(w,np.abs(chiR),c="red",label="R bilayer")
plt.plot(w,np.abs(chiV),c="blue",label="2H bilayer, V=0.1 eV")
plt.plot(w,np.abs(chi2H),c="gray",ls="--",label="2H bilayer")
plt.xlabel("Frequency [eV]") ; plt.ylabel("$|\\chi_{yyy}|$ [nm$^2$/V]")
plt.legend()
plt.tight_layout()
plt.show()
