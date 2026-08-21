# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# The Bethe-Salpeter equation on a 2D k-mesh, without the dense matrix.
#
# In two dimensions the dense wall arrives fast: N_pair = n_v n_c nk^2, so
# a 32x32 mesh on a two-band model already needs a 1024x1024 matrix and a
# 64x64 mesh a 4096x4096 one -- and a Wannier-Mott exciton in a 2D
# semiconductor is exactly the case that needs a fine mesh, because its
# envelope A(k) is squeezed into a small neighbourhood of the band edge.
#
# The 1D example (examples/1d/exciton_qtt) explains the two matrix-free
# solvers. What is different in 2D:
#
#   - the k index has bits from BOTH reciprocal directions, and how they
#     are ordered along the tensor train matters. `unfolding="grouped"`
#     (all kx bits, then all ky bits) is the default; `"interleaved"`
#     (by scale) is the textbook recommendation for multivariate quantics
#     functions and gives a HIGHER tensor-train rank on every 2D model
#     tried here (dE(k) at 128x128, tolerance 1e-6: rank 16 grouped
#     against 25 interleaved). Rank is not wall time, though: at the small
#     meshes below the two run about equally fast, and either can come out
#     ahead. The energies must agree exactly -- the bit ordering is a
#     representation choice, not a physical one -- and that is the point
#     of printing both.
#   - the exciton envelope is a 2D function, so it can be plotted over the
#     Brillouin zone, which is what the amplitudes are for.

import time
import numpy as np
import matplotlib.pyplot as plt
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction

# a gapped 2D semiconductor: honeycomb with a sublattice imbalance.
# Spinless, so the two bands are non-degenerate and the tensor-train ranks
# stay small -- a spinful model describes the same physics twice over and
# costs roughly four times the bond dimension for it.
h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
h.add_sublattice_imbalance(0.8)
h = h.get_multicell().get_dense()

def coulomb(e2):
    """A Coulomb tail with a soft cutoff at short distance"""
    return lambda r1,r2: e2/np.sqrt((r1-r2).dot(r1-r2)+0.25)

V = density_interaction(h,V1=0.6,Vr=coulomb(0.4)) # long ranged, so the
# exciton is shallow and the mesh actually matters

print("%8s %16s %16s %16s"%("nk","dense","iterative","qtt"))
for nk in [8,16]:
    e_dense = h.get_bse(V=V,nk=nk,tda=True).get_energies(n=1)[0].real
    e_iter = h.get_bse(V=V,nk=nk,tda=True,solver="iterative",
            neig=1).get_energies()[0].real
    e_qtt = h.get_bse(V=V,nk=nk,tda=True,solver="qtt",neig=1,
            tolerance=1e-8,coarse_nk=4).get_energies()[0].real
    print("%8s %16.10f %16.10f %16.10f"%("%dx%d"%(nk,nk),e_dense,e_iter,
        e_qtt))

# the bit ordering, measured rather than assumed
print()
print("%8s %10s %14s %10s %8s"%("nk","unfolding","E_qtt","k-points","time"))
for nk in [8,16]:
    for unf in ["grouped","interleaved"]:
        t0 = time.time()
        b = h.get_bse(V=V,nk=nk,tda=True,solver="qtt",neig=1,
                tolerance=1e-8,coarse_nk=4,unfolding=unf,maxdim=60)
        print("%8s %10s %14.10f %10d %7.1fs"%("%dx%d"%(nk,nk),unf,
            b.get_energies()[0].real,b.pairs.ndiag(),time.time()-t0))

# The exciton envelope over the Brillouin zone. The amplitudes come back
# indexed by the flat pair index, whose k-point is index // nband, so a
# 2D model reshapes straight into (nk,nk).
#
# The meshes here are deliberately modest. Two dimensions costs far more
# per mesh point than one: the MPO bond dimension is ~63 against ~8 for
# the 1D chain, and it enters the DMRG cost multiplied by the square of
# the MPS bond dimension. Finer 2D meshes do work -- raise nk (a power of
# two) and maxdim together -- they are simply not something to put in an
# example meant to run in a couple of minutes. solver="iterative" is the
# cheaper route to a fine 2D mesh when only the energy is wanted.
nk = 16
b = h.get_bse(V=V,nk=nk,tda=True,solver="qtt",neig=1,tolerance=1e-8,
        coarse_nk=4,maxdim=60)
A = b.amplitudes[0]
nband = b.pairs.nband
env = np.zeros(nk*nk)
for m in range(len(A)):
    env[m//nband] += abs(A[m])**2
env = env.reshape(nk,nk)
# roll so the zone center sits in the middle of the plot
env = np.roll(np.roll(env,nk//2,axis=0),nk//2,axis=1)

plt.figure(figsize=(5.5,4.5))
plt.imshow(env.T,origin="lower",extent=[-0.5,0.5,-0.5,0.5],
        aspect="equal",cmap="magma")
plt.colorbar(label=r"$|A(k)|^2$")
plt.xlabel(r"$k_1$") ; plt.ylabel(r"$k_2$")
plt.title("lowest exciton envelope, nk = %dx%d"%(nk,nk))
plt.tight_layout()
plt.show()
