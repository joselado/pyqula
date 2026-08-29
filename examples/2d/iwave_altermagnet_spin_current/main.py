# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import specialhamiltonian

## The i-wave altermagnet, and why it needs fifth order.
##
## An i-wave form factor is a k-space harmonic of order six, so the first
## derivative of the band energy that survives the Brillouin-zone integral
## is the sixth -- which is the FIFTH order in the electric field. Nothing
## below it responds at all. This is also why Ezawa's second-order CHARGE
## response (arXiv:2409.09241) cannot see an i-wave altermagnet: that one
## is keyed to the quadratic d-wave form factor.
##
## Ezawa, Phys. Rev. B 111, 125420 (2025), arXiv:2411.16036.

h = specialhamiltonian.iwave_altermagnet(J=0.3, t=-1.)

# the triangular band with t = -1 runs over [-6,3]; a chemical potential a
# little above the bottom puts a small Fermi pocket around Gamma
mu = -5.5

orders = h.get_nonlinear_drude_orders(lmax=6, nk=48, T=0.02, mu=mu)
for (l, v) in enumerate(orders):
    print(f"  l = {l}:  max |sigma_spin| = {v:.2e}")
print()

## The components at fifth order obey the relations Ezawa derives,
## sigma^{yyyyy;x} = sigma^{xxxxx;y} = -sigma^{xxxyy;y}, which hold on the
## lattice as well as in the continuum.
c = h.get_nonlinear_drude_components(5, nk=48, T=0.02, mu=mu)
ref = c["yyyyy;x"]
for key in ["yyyyy;x", "xxxxx;y", "xxxyy;y"]:
    print(f"  sigma^{{{key}}} / sigma^{{yyyyy;x}} = {np.real(c[key]/ref):+.4f}")
print()

## A TRAP worth knowing about. With the chemical potential in a gap, f is 1
## on every valence band and 0 on every conduction band, so the integrand
## is a pure k-derivative of Tr(P H) with P the valence projector. That is
## smooth and periodic, and the zone integral of a derivative of a smooth
## periodic function vanishes -- so an INSULATOR returns exactly zero at
## every order, the fifth included.
##
## So "nothing below fifth order" on its own does not identify i-wave
## order: a plain band insulator reproduces the same wall of zeros. The
## fingerprint is the CONTRAST -- zeros below fifth order together with a
## clearly nonzero fifth order -- and that requires a metallic state.

## (Below the band bottom the system is simply empty, which is the same
## conclusion reached the easy way.)
print("  with mu below the band bottom, nothing is occupied:")
print(f"    l = 5:  {h.get_nonlinear_drude_orders(lmax=5, nk=24, T=0.02, mu=-8.)[5]:.2e}")
print("  the filled-band case is checked in")
print("  tests/nonlinearconductivity/test_nonlinear_drude.py::"
      "test_a_gapped_insulator_gives_exactly_zero_at_every_order")

## The response is linear in the altermagnetic order parameter J over a
## wide range, so its size measures the order as well as its symmetry.
Js = np.linspace(0., 0.5, 11)
sig = [abs(specialhamiltonian.iwave_altermagnet(J=J, t=-1.).
           get_nonlinear_drude_conductivity(field="yyyyy", current="x",
               nk=48, T=0.02, mu=mu)) for J in Js]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].semilogy(range(7), np.maximum(orders, 1e-18), "o-", c="C3")
ax[0].set_xlabel("order l of the electric field")
ax[0].set_ylabel(r"max $|\sigma_{\rm spin}|$")
ax[0].set_title("i-wave: silent below l = 5")
ax[1].plot(Js, sig, "o-", c="C0")
ax[1].set_xlabel("altermagnetic order J")
ax[1].set_ylabel(r"$|\sigma_{\rm spin}^{yyyyy;x}|$")
ax[1].set_title("linear in J")
fig.tight_layout()
plt.show()
