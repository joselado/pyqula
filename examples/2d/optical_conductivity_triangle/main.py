# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Optical absorption with a very small broadening. A plain sum over the
# k-mesh gives one narrow Lorentzian per k-point, so with delta = 1e-3 t it
# shows the single k-points as spikes. The triangle integration takes the
# transition energies as linear inside each triangle of the mesh and
# integrates the resolvent exactly, so the same mesh gives a smooth curve.
# The qtci integration does the same on a finer mesh, whose transition
# energies and weights are interpolated from a small set of k-points.
# Units of e^2/hbar, so one conductance quantum e^2/h is 1/(2*pi)

import numpy as np
from pyqula import geometry

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.add_sublattice_imbalance(0.3) # gapped honeycomb, gap 0.6

ws = np.linspace(0.,3.5,351) # frequencies
kw = dict(energies=ws,nk=256,T=1e-3,delta=1e-3)
(ws,sg) = h.get_optical_conductivity(integration="grid",**kw) # spikes
(ws,st) = h.get_optical_conductivity(integration="triangle",**kw) # smooth
# finer mesh of 1024 x 1024 points, only a part of it diagonalized
(ws,sq) = h.get_optical_conductivity(energies=ws,nk=1024,T=1e-3,delta=1e-3,
                                     integration="qtci")

import matplotlib.pyplot as plt
plt.plot(ws,2*np.pi*sg[:,0,0].real,c="gray",lw=0.8,label="grid, nk=256")
plt.plot(ws,2*np.pi*st[:,0,0].real,c="blue",label="triangle, nk=256")
plt.plot(ws,2*np.pi*sq[:,0,0].real,c="red",ls="--",label="qtci, nk=1024")
plt.xlabel("Frequency [t]") ; plt.ylabel("Re $\\sigma_{xx}$ [$e^2/h$]")
plt.legend()
plt.show()
