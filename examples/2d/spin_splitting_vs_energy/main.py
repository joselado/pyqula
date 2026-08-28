# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import specialhamiltonian

## Energy-resolved spin splitting of a d-wave altermagnet, both ways.
##
## get_spin_splitting_density broadens every band pair into a smooth
## weighted density -- the TYPICAL splitting at each energy.
## get_spin_splitting_vs_energy keeps the LARGEST |Delta| in each bin, so
## its global maximum bounds the splitting anywhere in the Brillouin
## zone. That bound is the number to quote for a material, since a
## maximum taken along a single cut through k-space only answers for the
## cut you happened to pick.

h = specialhamiltonian.square_altermagnet(am=1.)

(es,ds) = h.get_spin_splitting_vs_energy(nk=100,nbins=400)
(xs,ys) = h.get_spin_splitting_density(nk=100,delta=1e-1,energies=es)

# for this model the answer is analytic: the maximum is exactly 4*am
print("largest spin splitting anywhere in the BZ:",ds.max())

## The unit cell has to be the true magnetic one. Bands are paired by
## sorted index, and a supercell folds several k-points onto each k, so
## the pairing then compares bands that came from different momenta and
## the maximum comes out too small -- silently. Here it halves:
hs = specialhamiltonian.square_altermagnet(am=1.).supercell(2)
print("same system in a redundant 2x2 cell:",
      hs.get_spin_splitting_vs_energy(nk=50,nbins=400)[1].max())

import matplotlib.pyplot as plt

fig,ax = plt.subplots(2,1,sharex=True,figsize=(6,6))
ax[0].plot(es,ds,c="C3")
ax[0].set_ylabel("max spin splitting")
ax[1].plot(xs,ys,c="C0")
ax[1].set_ylabel("spin splitting density")
ax[1].set_xlabel("energy")
plt.tight_layout()
plt.show()
