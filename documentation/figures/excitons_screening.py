import numpy as np
import style
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction

h = geometry.honeycomb_lattice().get_hamiltonian()
h.add_sublattice_imbalance(1.0) # a gapped semiconductor

# a BARE interaction: an onsite term plus a soft-cutoff Coulomb tail
coulomb = lambda r1,r2: 0.6/np.sqrt((r1-r2).dot(r1-r2)+0.25)
V = density_interaction(h,U=1.0,Vr=coulomb)

bare = h.get_bse(V=V,nk=8) # time-dependent Hartree-Fock
screened = h.get_bse(V=V,nk=8,screening="rpa") # GW-BSE style

W = h.get_screened_interaction(V=V,nk=8)
print(W.epsmin) # smallest dielectric eigenvalue over the mesh
d = W.get_dict() # ... and back in real space, usable at any q

# the interaction between opposite spins on two sites, against their distance
r = h.geometry.r # positions of the two sites of the cell
a1, a2 = h.geometry.a1, h.geometry.a2
dist, v, w = [], [], []
for (n1,n2,n3) in V:
    for i in range(2):
        for j in range(2):
            dist.append(np.linalg.norm(r[j] + n1*a1 + n2*a2 - r[i]))
            v.append(V[(n1,n2,n3)][2*i,2*j+1].real) # spin up on i, down on j
            w.append(d[(n1,n2,n3)][2*i,2*j+1].real)

def draw(ax, c):
    for (x, b, label) in [(0, bare, "bare"), (1, screened, "screened")]:
        for e in b.get_energies()[:12]:
            ax[0].hlines(e, x-0.3, x+0.3, color=c.blue, lw=2)
        ax[0].text(x, b.get_energies()[0]-0.02, "binding %.3f" % b.get_binding_energies()[0],
                   color=c.ink2, ha="center", va="top")
    ax[0].axhline(bare.get_lowest_transition(), color=c.ink2, ls="--", lw=1)
    ax[0].text(1.45, bare.get_lowest_transition()+0.01, "lowest free pair",
               color=c.ink2, ha="right", va="bottom")
    ax[0].set_xticks([0, 1], ["bare", "screened"])
    ax[0].set_xlim(-0.5, 1.5)
    ax[0].set_ylim(1.85, 2.25)
    ax[0].set_ylabel(r"exciton energy [$t$]")
    ax[1].scatter(dist, v, s=60, facecolors="none", edgecolors=c.ink2,
                  lw=1.5, label="bare $v$")
    ax[1].scatter(dist, w, s=20, color=c.orange, label="screened $W$")
    ax[1].set_xlim(-0.2, 5.1)
    ax[1].set_xlabel("distance [first-neighbor distance]")
    ax[1].set_ylabel(r"interaction [$t$]")
    ax[1].legend(loc="upper right")

style.save(draw, "excitons_screening", npanels=2)
