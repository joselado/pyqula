# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Nonlinear Hall effect of a doped gapped honeycomb lattice under strain.
# A dc field E gives a current at second order, J_a = sigma_abc E_b E_c.
# With time reversal the only term is the Berry curvature dipole,
#   D_a = sum_k sum_n (-df/dE) v^n_a Omega^n_xy / (N_k V_cell),
# with sigma_xyy = tau D_y and sigma_yxx = -tau D_x, and tau the
# relaxation time. The C3 rotation of the unstrained lattice makes D
# zero, so the uniaxial strain is what switches the effect on.
# The chemical potential is put inside the conduction band with
# h.shift_fermi(-mu), and T has to be above the energy spacing of the mesh.

import numpy as np
from pyqula import geometry, strain

def get_h(mu,s):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False) # spinless, one spin species
    h.add_sublattice_imbalance(0.5) # staggered onsite energy, gap 1.0
    if s!=0.: strain.uniaxial_strain(h,d=np.array([1.,0.4,0.]),s=s) # breaks C3
    h.shift_fermi(-mu) # chemical potential at mu
    return h

mus = np.linspace(0.3,2.5,45) # chemical potentials, the gap edge is at 0.5
D = np.array([get_h(mu,0.15).get_berry_curvature_dipole(nk=150,T=0.05) for mu in mus])
D0 = np.array([get_h(mu,0.).get_berry_curvature_dipole(nk=150,T=0.05) for mu in mus])
print("largest |D| with strain   ",np.max(np.abs(D)))
print("largest |D| without strain",np.max(np.abs(D0))) # zero by C3

# the three terms of the second-order dc response at one chemical potential,
# here with a Haldane term too, so that time reversal is broken
h = get_h(0.9,0.15)
h.add_haldane(0.1)
t = h.get_nonlinear_dc_conductivity(nk=150,T=0.05,tau=1.,components=["yxx","xyy"])
for term in ["drude","bcd","qmd","total"]: # tau^2, tau^1, tau^0 and the sum
    print(term,"sigma_yxx = %.4e   sigma_xyy = %.4e"%(t[term]["yxx"],t[term]["xyy"]))

import matplotlib.pyplot as plt

plt.plot(mus,D[:,0],c="blue",label="$D_x$, strained")
plt.plot(mus,D[:,1],c="red",label="$D_y$, strained")
plt.plot(mus,D0[:,0],c="black",ls="--",label="$D_x$, no strain")
plt.axvline(0.5,c="gray",ls=":") # conduction band edge
plt.xlabel("Chemical potential [t]") ; plt.ylabel("Berry curvature dipole")
plt.legend()
plt.tight_layout()
plt.show()
