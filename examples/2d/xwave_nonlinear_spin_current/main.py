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
##   d-wave: l = 1   f-wave: l = 2   g-wave: l = 3   i-wave: l = 5
##
## Reading off the LOWEST order at which a nonlinear spin current appears
## therefore identifies the wave index. Orders above the threshold are
## generically nonzero too on a lattice, because the lattice form factor
## carries higher harmonics beyond the leading one -- it is the absence of
## everything below the threshold that carries the information.
##
## The p-wave magnet (form factor kx) has NO spin current at any order.
## 2t cos kx + s J sin kx = sqrt(4t^2+J^2) cos(kx - s phi) is the same band for
## both spins, shifted rigidly along kx by opposite amounts, and a zone
## integral does not see a rigid shift. The l = 0 column is zero for every
## wave anyway: f d eps/dk_b = d F(eps)/dk_b is a total derivative, so there
## is no current in equilibrium.
##
## Ezawa, Phys. Rev. B 111, 125420 (2025), arXiv:2411.16036 (Sec. V for the
## p-wave magnet).

WAVES = ["p","d","f","g","i"]
THRESHOLD = {"d":1,"f":2,"g":3,"i":5} # the p-wave magnet has none
BOTTOM = {"p":-4.,"d":-4.,"f":-6.,"g":-4.,"i":-6.} # band bottom for t = -1
ZERO = 1e-10 # anything below this is a zero

## The mesh matters for the p-wave row. Its rigid shift is not a translation
## of the k-mesh, so its zero is reached only as nk grows (exponentially at
## finite T): its even orders are 3e-4, 2e-5, 6e-8 and 1e-11 at nk = 48, 96,
## 192 and 384, and at nk = 48 it looks exactly like a response at l = 0. The
## zeros below the threshold of the other four are forced by symmetry and are
## machine zeros on any mesh. nk = 384 also converges the i-wave l = 5 entry,
## which is 2.5% low at nk = 48.
table = dict()
for wave in WAVES:
    h = specialhamiltonian.xwave_magnet(wave=wave,J=0.3,t=-1.)
    # the largest |sigma^{x^l1 y^l2 ; b}| over every component of order l
    table[wave] = h.get_nonlinear_drude_orders(lmax=6,nk=384,T=0.02,
            mu=BOTTOM[wave]+0.5)

print("largest |sigma_spin| over the components of each order\n")
print("wave |"+"".join(f"    l={l}  " for l in range(7)))
for wave in WAVES:
    print(f"  {wave}  |"+"".join(f" {v:8.1e}" for v in table[wave]))
print()
## Read off the wave index. Note the guard on "nothing is nonzero": argmax
## on an all-below-threshold array silently returns 0, which would report a
## response at l = 0, an order at which no system responds, for a system
## with no spin response at all. "No response at any order" is the right
## answer for the p-wave magnet, and it is also what a spin-DEGENERATE state
## (a Neel antiferromagnet, not an altermagnet) gives; the second one is
## flagged with a warning by the spin-splitting guard.
for wave in WAVES:
    above = np.where(np.array(table[wave]) > ZERO)[0]
    expected = THRESHOLD.get(wave,"none")
    if len(above) == 0:
        print(f"  {wave}-wave: no response at any order  (expected {expected})")
    else:
        print(f"  {wave}-wave: response starts at l = {above[0]}"
              f"  (expected {expected})")

## The i-wave row is the point of the exercise. Orders 0 through 4 are zero
## to machine precision -- not small, zero -- and the response appears only
## at fifth order. This is why Ezawa's SECOND-order charge response
## (arXiv:2409.09241) cannot detect an i-wave altermagnet at all: it is
## keyed to the quadratic d-wave form factor.

import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize=(6.5,4.2))
for (i,wave) in enumerate(WAVES):
    y = np.array(table[wave])
    # every zero drawn at the same floor, so that they are visible on a log
    # axis and a p-wave mesh error cannot stand out above the other zeros
    ax.semilogy(range(7),np.where(y<ZERO,ZERO,y),"o-",c="C%d"%i,
            label=wave+"-wave"+("" if wave in THRESHOLD else " (none)"))
    if wave in THRESHOLD:
        ax.axvline(THRESHOLD[wave],ls=":",lw=0.8,c="C%d"%i)
ax.set_xlabel("order l of the electric field")
ax.set_ylabel(r"max $|\sigma_{\rm spin}^{x^{l_1}y^{l_2};b}|$ (zeros at %.0e)"%ZERO)
ax.set_title("nonlinear spin current switches on at the X-wave index")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
