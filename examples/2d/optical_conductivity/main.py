# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Optical (Kubo-Greenwood) conductivity of the Haldane model, a Chern
# insulator. Re sigma_xx is the optical absorption, which switches on at
# the direct gap, and Im sigma_xy is the magneto-optical response, whose
# zero-frequency limit is quantized: sigma_xy(0) = -C e^2/h with C the
# Chern number. Everything is in units of e^2/hbar, so one conductance
# quantum e^2/h is 1/(2*pi) = 0.159

import numpy as np
from pyqula import geometry
from pyqula import topology

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.add_haldane(0.2) # Haldane coupling, opens a topological gap
h.shift_fermi(0.3) # put the Fermi energy in the middle of the gap

energies = np.linspace(0.0,4.0,100) # frequencies
(ws,sigma) = h.get_optical_conductivity(energies=energies,nk=40,
                                        T=0.02,delta=0.05)

# the DC Hall conductivity, compared with the Chern number
(w0,s0) = h.get_optical_conductivity(energies=[0.],nk=40,T=0.01,delta=1e-3)
print("Chern number         ",topology.chern(h,nk=20))
print("sigma_xy(0) in e^2/h ",2.*np.pi*s0[0,0,1].real)

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(ws,sigma[:,0,0].real,c="blue")
plt.axvline(h.get_gap(),c="black",ls="--") # direct gap
plt.xlabel("Frequency") ; plt.ylabel("Re $\\sigma_{xx}$ [$e^2/\\hbar$]")

plt.subplot(1,2,2)
plt.plot(ws,sigma[:,0,1].real,c="red",label="Re")
plt.plot(ws,sigma[:,0,1].imag,c="green",label="Im")
plt.axhline(-1./(2.*np.pi),c="black",ls="--") # -1 conductance quantum
plt.xlabel("Frequency") ; plt.ylabel("$\\sigma_{xy}$ [$e^2/\\hbar$]")
plt.legend()

plt.tight_layout()
plt.show()
