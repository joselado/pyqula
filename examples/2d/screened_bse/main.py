# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# The screened interaction, computed after the mean field, and the BSE
# built on top of it.
#
# The default BSE uses the bare interaction in both of its terms, which
# makes it time-dependent Hartree-Fock. But an electron and a hole added
# to a solid do not feel the bare interaction: the other electrons
# rearrange around them, and what survives is the screened interaction
#
#     W(q) = eps^-1(q) v(q),   eps(q) = 1 - v(q) chi0(q)
#
# with chi0 the static polarizability of the very bands the mean field
# produced. screening="rpa" computes it and puts it in the direct (ladder)
# term, leaving the exchange term bare -- the standard GW-BSE split.
#
# eps is built in the CHARGE channel, on site indices, as in GW: what
# polarizes the medium is the total density, and the correction that comes
# back is spin independent. That is what keeps the result spin-rotation
# invariant -- dressing the full spin-orbital matrix instead
# (channel="orbital") generates an Ising Sz-Sz coupling and splits the
# exciton's spin multiplet, which this example also shows.
#
# Below: the dielectric matrix across the Brillouin zone, and what
# screening does to the exciton.

import numpy as np
import matplotlib.pyplot as plt
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction, interaction_at_q

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_sublattice_imbalance(1.0) # open a gap, a two-band semiconductor

def coulomb(e2):
    """A Coulomb tail with a soft cutoff at short distance"""
    return lambda r1,r2: e2/np.sqrt((r1-r2).dot(r1-r2)+0.25)

# NOTE this is a BARE interaction, which is what may be screened. A
# Hubbard U fitted to a material is already an effective screened
# interaction and must NOT be run through this a second time.
V = density_interaction(h,U=1.0,Vr=coulomb(0.6))

nk = 8
W = h.get_screened_interaction(V=V,nk=nk) # the screened interaction
print(W)

# how strongly each q-point is screened: the eigenvalues of eps(q)
qs,eigs = [],[]
for iq,q in enumerate(W.qs):
    vq = interaction_at_q(W.bare,g,q) # the bare interaction at this q
    eps = np.identity(vq.shape[0]) - vq@W.chi0[iq] # dielectric matrix
    qs.append(q) ; eigs.append(np.sort(np.linalg.eigvals(eps).real))
eigs = np.array(eigs)

# the screened interaction back in real space, to see how far it reaches.
# get_dict() inverse Fourier transforms the tabulated W(q); the result is
# usable at any q (unlike W itself, which lives only on the mesh)
d = W.get_dict()
strength = sorted(((np.max(np.abs(m)),k) for k,m in d.items()),reverse=True)
print("\nscreened interaction in real space, strongest cells:")
for val,k in strength[0:6]:
    print("   cell %-12s  %.4f"%(str(k),val))

# and what it does to the exciton
print("\n%-28s %10s %10s   %s"%("","lowest","binding","lowest 4 (multiplet)"))
for label,opts in [("bare",dict()),
                   ("screened, charge channel",dict(screening="rpa")),
                   ("screened, orbital channel",dict(screening="rpa",
                                                     channel="orbital"))]:
    b = h.get_bse(V=V,nk=nk,**opts)
    es = b.get_energies(4).real
    print("%-28s %10.6f %10.6f   spread %.1e"%(label,es[0],
            b.get_binding_energies(1)[0].real,es[3]-es[0]))
print("\nThe multiplet stays degenerate in the charge channel and splits in")
print("the orbital one: that split is the SU(2)-breaking artifact.")

plt.plot(np.sort(eigs.ravel()),marker=".",ls="")
plt.axhline(1.,color="k",ls="--",label="unscreened")
plt.xlabel("Eigenvalue of eps(q), sorted over the mesh")
plt.ylabel("Dielectric eigenvalue")
plt.title("Screening channel by channel")
plt.legend() ; plt.tight_layout()
plt.show()
