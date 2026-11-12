# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Tunneling spectrum (STM dI/dV) of a spinon mean field with a Kondo peak.
#
# A tip above a site of a spin model on a metallic substrate measures the
# spin flips of that site. The electron that tunnels exchanges its spin
# with the site, with a dimensionless coupling Jrho_s to the electrons of
# the substrate. To second order in the exchange, each spin excitation of
# energy w opens a step of dI/dV at eV = w, so dI/dV is the integral of
# the local spin spectrum up to the bias. To third order a logarithm
# ln(omega0/|eV - w|) appears at each step. For a spin that can flip at no
# cost the logarithm is at zero bias: this is the Kondo peak. The formulas
# are those of Ternes, arXiv:1505.04430, and here the spin correlators are
# those of the spinons, S_i = 1/2 f_i^dagger sigma f_i.
#
# We look at three cases:
#  1) one site without exchange, compared with the spin 1/2 of the paper;
#  2) the uniform Heisenberg chain, a continuum of spinons;
#  3) a chain of trimers, where each trimer has a spinon level at the
#     Fermi energy.

import numpy as np
from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian

np.random.seed(0)

# 1) One site without exchange is a spinon level at the Fermi energy with
# one fermion. We use the parameters of Fig. 7b of the paper (arXiv
# version), with the energies in meV: T = 1 K, a band cutoff of 20 meV and
# Jrho_s = -0.05. The free fermion can also be empty or doubly occupied,
# and these two states carry no spin, so its spectrum is half of the one
# of a spin 1/2. Twice the spectrum is the curve of the paper.
g = geometry.chain()
g.dimensionality = 0 # one isolated site
h = SpinonHamiltonian(g)
kT = 8.617333262e-2 # 1 K in meV
(e,d) = h.get_kondo_spectrum(np.array([0.,4.]),site=0,Jrho_s=-0.05,T=kT,
        omega0=20.,Gamma0=5e-3)
print("One site: twice dI/dV over 2 pi at zero bias and at 4 mV")
print(2*d/(2*np.pi),"(the paper has 1.13 and 0.886)")

# 2) The uniform chain. The spinons have a band that crosses the Fermi
# energy, so the spin spectrum grows linearly from zero energy and dI/dV
# starts as the square of the bias. There is no spin flip at zero energy
# and the third order adds no peak.
energies = np.linspace(-1.5,1.5,301)
T = 0.005 # temperature of the tip and the substrate, in units of J
h = SpinonHamiltonian(geometry.chain())
h2 = h.get_mean_field_hamiltonian(J1=1.0,decoupling="bond",nk=60,T=0.01)
(e,dchain) = h2.get_kondo_spectrum(energies,site=0,Jrho_s=-0.05,T=T,
        omega0=2.0,nk=400)
i0 = np.argmin(np.abs(e)) # zero bias
i1 = np.argmin(np.abs(e-0.1))
print("Uniform chain: dI/dV at zero bias and at 0.1 J")
print(dchain[i0],dchain[i1])

# 3) A chain of trimers: exchange 1 inside a trimer and 0.1 between two
# trimers. A trimer has three spinons in the levels -sqrt(2) t, 0 and
# sqrt(2) t, so the middle level holds one spinon and it can flip at no
# cost. That level lives on the two end sites, and the middle site has no
# weight in it. The Kondo peak then appears on the end sites only.
g = geometry.chain().get_supercell(3) # the sites are at x = -1, 0, 1
def J(r1,r2):
    if abs(np.linalg.norm(r1-r2)-1.)>1e-3: return 0. # first neighbors only
    inside = abs((r1[0]+r2[0])/2.)<1. # a bond inside the cell
    return 1.0 if inside else 0.1
h = SpinonHamiltonian(g)
h2 = h.get_mean_field_hamiltonian(Jr=J,rcut=1.5,decoupling="bond",
        ansatz=g.get_hamiltonian(has_spin=True),hold="phases",nk=60,T=0.01)
(e,d2) = h2.get_kondo_spectrum(energies,order=2,T=T,omega0=2.0,nk=400)
(e,d3) = h2.get_kondo_spectrum(energies,Jrho_s=-0.05,T=T,omega0=2.0,nk=400)
print("Trimers: dI/dV at zero bias and at 0.1 J, for the three sites")
for i in range(3):
    print(" site",i,"second order",d2[i,i0],d2[i,i1],
          "with the Kondo term",d3[i,i0],d3[i,i1])

np.savetxt("KONDO_SPECTRUM.OUT",np.array([e,dchain,d3[0],d3[1],d3[2]]).T)
