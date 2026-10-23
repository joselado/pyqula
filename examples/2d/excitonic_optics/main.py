# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Optical absorption and second-harmonic generation with excitons, from
# the Bethe-Salpeter equation in the Tamm-Dancoff approximation. A gapped
# honeycomb lattice (gap 2t) with a Coulomb tail binds an exciton at
# 1.845t. It gives an absorption peak below the gap, and two peaks in the
# second harmonic: at half the exciton energy (2w resonance) and at the
# exciton energy (w resonance). Without the interaction (kernel="none")
# the same functions give the results without excitons.

import numpy as np
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False) # spinless, one spin species
h.add_sublattice_imbalance(1.0) # gap 2t at K and K'
W = density_interaction(h,Vr=lambda r1,r2: 0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25))
zero = {(0,0,0):np.zeros(h.intra.shape,dtype=np.complex128)}

kw = dict(energies=np.linspace(0.,2.6,131),nk=24,delta=0.04)
(ws,s) = h.get_excitonic_optical_conductivity(V=W,component="xx",**kw)
(ws,s0) = h.get_excitonic_optical_conductivity(V=zero,kernel="none",
                                               component="xx",**kw)
(ws,chi) = h.get_excitonic_shg_susceptibility(V=W,components="xxx",**kw)
(ws,chi0) = h.get_excitonic_shg_susceptibility(V=zero,kernel="none",
                                               components="xxx",**kw)
print("absorption maximum at ",ws[np.argmax(s.real)]) # the exciton
print("SHG maximum at        ",ws[np.argmax(np.abs(chi))]) # half of it

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(ws,2*np.pi*s.real,c="blue",label="excitons")
plt.plot(ws,2*np.pi*s0.real,c="red",label="no interaction")
plt.xlabel("Frequency [t]") ; plt.ylabel("Re $\\sigma_{xx}$ [$e^2/h$]")
plt.legend()
plt.subplot(1,2,2)
plt.plot(ws,np.abs(chi),c="blue",label="excitons")
plt.plot(ws,np.abs(chi0),c="red",label="no interaction")
plt.xlabel("Frequency [t]") ; plt.ylabel("$|\\chi_{xxx}|$ [nm$^2$/V]")
plt.legend()
plt.tight_layout()
plt.show()
