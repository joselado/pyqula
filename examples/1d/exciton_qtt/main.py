# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Solving the Bethe-Salpeter equation without ever building its matrix.
#
# The dense BSE matrix is (nv*nc*nk)^2, so the k-mesh -- the knob that
# actually controls whether an exciton is converged -- is limited by
# memory. A Wannier-Mott exciton whose envelope A(k) is sharp in the
# Brillouin zone needs a mesh far denser than a tightly bound one, and
# that is exactly what the dense solver cannot give.
#
# Two solvers avoid the matrix entirely:
#
#   solver="iterative"  the BSE kernel is EXACTLY a diagonal plus a fixed
#       number of rank-one terms -- one per non-zero entry of the
#       real-space interaction, independent of nk -- so it can be applied
#       to a vector in O(nk) work and handed to a Lanczos eigensolver. No
#       matrix, no memory wall. Still O(nk) per iteration.
#
#   solver="qtt"  binary-encode the pair index, cross-interpolate the
#       kernel into a matrix product operator with qutecipy, and
#       diagonalize it with DMRG (pyitensor, from dmrgpy). The number of
#       k-points ever diagonalized then grows like log(nk) instead of nk.
#
# Both are Tamm-Dancoff (tda=True); the full non-Tamm-Dancoff problem
# needs the dense solver.
#
# This example runs the same physics at meshes spanning three orders of
# magnitude and shows (a) that the three solvers agree where they overlap
# and (b) that the quantics one keeps working, at nearly constant cost,
# far past where the others stop.

import time
import numpy as np
import matplotlib.pyplot as plt
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction

# a gapped, spinless two-band chain: a staggered onsite potential on a
# two-site cell. Spinless keeps the bands non-degenerate and the
# tensor-train ranks small; see below for what degeneracy costs.
g = geometry.chain().supercell(2)
h = g.get_hamiltonian(has_spin=False)
h.add_onsite(lambda r: 0.9*(-1)**int(round(r[0]-0.5)))
h = h.get_multicell().get_dense()

V = density_interaction(h,V1=0.8) # nearest-neighbor repulsion

print("solver comparison, lowest exciton energy")
print("%8s %16s %16s %16s"%("nk","dense","iterative","qtt"))
rows = []
for nk in [8,16,32,64,128,256]:
    e_dense = h.get_bse(V=V,nk=nk,tda=True).get_energies(n=1)[0].real
    e_iter = h.get_bse(V=V,nk=nk,tda=True,solver="iterative",
            neig=1).get_energies()[0].real
    e_qtt = h.get_bse(V=V,nk=nk,tda=True,solver="qtt",neig=1,
            tolerance=1e-10).get_energies()[0].real
    print("%8d %16.10f %16.10f %16.10f"%(nk,e_dense,e_iter,e_qtt))
    rows.append((nk,e_dense,e_iter,e_qtt))

# Past the dense wall. The quantics solver reports how many k-points it
# actually diagonalized -- that number, not nk, is what it costs.
print()
print("%8s %16s %10s %10s %8s"%("nk","E_qtt","npair","k-points","time"))
big = []
for nk in [1024,4096,16384,65536,262144]:
    t0 = time.time()
    b = h.get_bse(V=V,nk=nk,tda=True,solver="qtt",neig=1,tolerance=1e-8)
    dt = time.time()-t0
    e = b.get_energies()[0].real
    print("%8d %16.10f %10d %10d %7.1fs"%(nk,e,b.pairs.npair,
        b.pairs.ndiag(),dt))
    big.append((nk,e,b.pairs.ndiag(),dt))

fig,ax = plt.subplots(1,2,figsize=(11,4))
nks = [r[0] for r in rows]
ax[0].plot(nks,[r[1] for r in rows],"o-",label="dense")
ax[0].plot(nks,[r[2] for r in rows],"s--",label="iterative")
ax[0].plot(nks,[r[3] for r in rows],"^:",label="qtt")
ax[0].plot([r[0] for r in big],[r[1] for r in big],"^:",color="C2")
ax[0].set_xscale("log",base=2)
ax[0].set_xlabel("nk") ; ax[0].set_ylabel("lowest exciton energy")
ax[0].legend() ; ax[0].set_title("the three solvers agree")

ax[1].plot([r[0] for r in big],[r[0] for r in big],"k--",
        label="every k-point")
ax[1].plot([r[0] for r in big],[r[2] for r in big],"^-",
        label="diagonalized by qtt")
ax[1].set_xscale("log",base=2) ; ax[1].set_yscale("log",base=2)
ax[1].set_xlabel("nk") ; ax[1].set_ylabel("k-points diagonalized")
ax[1].legend() ; ax[1].set_title("cost grows like log(nk)")
plt.tight_layout()
plt.show()

# WHAT MAKES IT WORK, and what breaks it.
#
# The Bloch eigenvectors come out of the diagonalization with an arbitrary
# phase at each k, so the kernel is a discontinuous function of k even
# where the physics is smooth -- and a discontinuous function has no
# low-rank quantics representation at all. The solver therefore fixes the
# gauge (bsetk/gauge.py) before encoding anything. The default,
# gauge="projection", rotates each band subspace onto fixed trial
# orbitals; it is the one that also works when bands are degenerate, where
# what is arbitrary is a whole unitary rather than a phase.
#
# The gauge changes no energy -- it is a unitary on the pair index -- so
# it can be switched on for the dense solver too, and the spectrum must
# not move:
nk = 16
raw = np.sort(h.get_bse(V=V,nk=nk,tda=True).get_energies().real)
fixed = np.sort(h.get_bse(V=V,nk=nk,tda=True,
    gauge="projection").get_energies().real)
print()
print("gauge fixing changes the spectrum by %.2e"%np.max(np.abs(raw-fixed)))
