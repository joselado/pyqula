# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import specialhamiltonian

## Measuring altermagnetic order electrically, with no spin-orbit coupling.
##
## An X-wave collinear magnet has a spin-splitting form factor that is a
## k-space harmonic of order l+1. Its (l+1)-th derivative is the first one
## that is a nonzero constant, and every lower one integrates to zero over
## the Brillouin zone, so the nonlinear Drude spin current switches on at
## order l and not before:
##
##   p-wave: l = 0   d-wave: l = 1   f-wave: l = 2
##   g-wave: l = 3   i-wave: l = 5
##
## Reading off the LOWEST order at which a nonlinear spin current appears
## therefore identifies the wave index. Orders above the threshold are
## generically nonzero too on a lattice, because the lattice form factor
## carries higher harmonics beyond the leading one -- it is the absence of
## everything below the threshold that carries the information.
##
## Ezawa, Phys. Rev. B 111, 125420 (2025), arXiv:2411.16036.

WAVES = ["p","d","f","g","i"]
THRESHOLD = {"p":0,"d":1,"f":2,"g":3,"i":5}
BOTTOM = {"p":-4.,"d":-4.,"f":-6.,"g":-4.,"i":-6.} # band bottom for t = -1

table = dict()
for wave in WAVES:
    h = specialhamiltonian.xwave_magnet(wave=wave,J=0.3,t=-1.)
    # the largest |sigma^{x^l1 y^l2 ; b}| over every component of order l
    table[wave] = h.get_nonlinear_drude_orders(lmax=6,nk=48,T=0.02,
            mu=BOTTOM[wave]+0.5)

print("largest |sigma_spin| over the components of each order\n")
print("wave |"+"".join(f"    l={l}  " for l in range(7)))
for wave in WAVES:
    print(f"  {wave}  |"+"".join(f" {v:8.1e}" for v in table[wave]))
print()
## Read off the wave index. Note the guard on "nothing is nonzero": argmax
## on an all-below-threshold array silently returns 0, which would report a
## persistent spin current for a system that simply has no spin response at
## all. That is not hypothetical -- it is exactly how a spin-DEGENERATE
## fixture (a Neel antiferromagnet, not an altermagnet) can be mistaken for
## a p-wave magnet.
for wave in WAVES:
    above = np.where(np.array(table[wave]) > 1e-9)[0]
    if len(above) == 0:
        print(f"  {wave}-wave: NO response at any order -- check that the "
              f"state is actually spin split")
    else:
        print(f"  {wave}-wave: response starts at l = {above[0]}"
              f"  (expected {THRESHOLD[wave]})")

## The i-wave row is the point of the exercise. Orders 0 through 4 are zero
## to machine precision -- not small, zero -- and the response appears only
## at fifth order. This is why Ezawa's SECOND-order charge response
## (arXiv:2409.09241) cannot detect an i-wave altermagnet at all: it is
## keyed to the quadratic d-wave form factor.

import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize=(6.5,4.2))
for (i,wave) in enumerate(WAVES):
    y = np.array(table[wave])
    # floor the machine zeros so they are visible on a log axis
    ax.semilogy(range(7),np.maximum(y,1e-18),"o-",label=wave+"-wave",c="C%d"%i)
    ax.axvline(THRESHOLD[wave],ls=":",lw=0.8,c="C%d"%i)
ax.set_xlabel("order l of the electric field")
ax.set_ylabel(r"max $|\sigma_{\rm spin}^{x^{l_1}y^{l_2};b}|$")
ax.set_title("nonlinear spin current switches on at the X-wave index")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
