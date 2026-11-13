# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Multipole chiral number N_xy of the quadrupole insulator of Benalcazar,
# Bernevig and Hughes. The hopping is 1-delta inside the unit cell and
# 1+delta between unit cells, so N_xy=1 for delta>0 and N_xy=0 for delta<0.
# N_xy is the number of zero modes at each corner of the open flake, all on
# one sublattice, which STM sees as a peak at zero bias at the corners.
import numpy as np
from pyqula import specialhamiltonian, algebra

L = 10 # sample of L x L unit cells
for delta in [-0.5,-0.2,0.2,0.5]:
    h = specialhamiltonian.square_2OTI(delta) # BBH model
    h = h.get_supercell((L,L)) # periodic sample
    r = h.geometry.r - h.geometry.r[0] # positions from the first site
    S = np.diag((-1.)**np.round(r[:,0]+r[:,1])) # sublattice, first site on A
    N = h.get_multipole_chiral_number(cells=(L,L),chiral=S) # on the torus
    No = h.get_multipole_chiral_number(cells=(L,L),chiral=S,boundary="open")
    # the zero modes of the open flake, on A minus on B, in the corner (0,0)
    (e,v) = np.linalg.eigh(algebra.todense(h.intra)) # levels of the flake
    v0 = v[:,np.abs(e)<1e-3] # states at zero energy
    w = np.sum(np.abs(v0)**2,axis=1)*np.diag(S) # weight times sublattice
    rc = h.geometry.r - h.geometry.r.min(axis=0) # from the corner of the flake
    corner = (rc[:,0]<L)&(rc[:,1]<L) # the quarter of the flake at (0,0)
    print("delta =",delta,"  N_xy torus =",round(N,6),"  N_xy open =",
          round(No,6),"  zero modes =",v0.shape[1],
          "  A-B at the corner =",round(np.sum(w[corner]),3))
# For delta=0.2 the corner modes decay slowly: the torus already gives 1,
# but their energy in the open flake is still 0.016, above the 1e-3 used
# here to count them, so the corner count needs a larger flake
