import numpy as np
import style
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

hc = geometry.chain().get_hamiltonian(has_spin=True) # conduction electrons
h = KondoLatticeHamiltonian(hc)

def J1(rc, rm): # Kondo coupling to the two first neighbors only
    if abs(np.linalg.norm(rc - rm) - 1.0) < 1e-6: return 2.0
    return 0.0

# onsite coupling J=4 (J'=2t), and the same J'/2 on each of the two neighbors
h0 = h.get_mean_field_hamiltonian(J=4.0, filling=0.5, nk=200, mf=(1.0, 0.0))
h1 = h.get_mean_field_hamiltonian(J=J1, filling=0.5, nk=200, mf=(1.0, 0.0))
print(h1.kondo_bonds)   # the bonds to the cells R=-1 and R=+1
print(h1.hybridization) # one field per bond

bands = []
for h2 in (h0, h1):
    (k, e, f) = h2.get_bands(operator="zposition") # weight on the f-sites
    bands.append((2*np.pi*k/k.max(), e, f))

def draw(ax, c):
    titles = ["onsite coupling", "coupling to the neighbors"]
    for a, (k, e, f), title in zip(ax, bands, titles):
        sc = a.scatter(k, e, c=2*f-1, s=6, cmap=c.diverging, vmin=-1, vmax=1)
        a.axhline(0, color=c.ink, lw=0.8, ls="--") # chemical potential
        style.kpath(a, k, ["0", r"$\pi$", r"$2\pi$"])
        a.set_xlabel("momentum $k$")
        a.set_title(title)
    ax[0].set_ylabel("energy [$t$]")
    style.colorbar(sc, ax[1], "conduction (-1) to f-site (+1)")

style.save(draw, "pseudofermions_kondobonds", npanels=2)
