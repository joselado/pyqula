import numpy as np
import style
from pyqula import geometry
from pyqula.symmetrytk import pointgroup

g = geometry.kagome_lattice()
h = g.get_hamiltonian(has_spin=False)

try:
    h.get_wannier_hamiltonian(bands=[0,0],nk=12,symmetries="auto") # the flat band alone
except ValueError as e:
    print("Flat band alone correctly rejected:",str(e).splitlines()[0])

# the full 3-band manifold has no such obstruction
hwan_sym = h.get_wannier_hamiltonian(bands=[0,2],nk=12,symmetries="auto")
print("Symmetries enforced:",[c.op.name for c in hwan_sym.wannier_symmetries])

(k,e) = h.get_bands(write=False) # bands of the kagome lattice
(kw,ew) = hwan_sym.get_bands(write=False) # bands of the Wannier model

def draw(ax, c):
    ax.scatter(k, e, s=30, color=c.blue, label="kagome lattice")
    ax.scatter(kw, ew, s=3, color=c.orange, label="Wannier model")
    ticks = np.array([0, 1/3, 1/2, 2/3, 1])*max(k) # the path goes through K, M and K'
    ax.set_xticks(ticks, [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"])
    ax.set_xlim(0, max(k))
    ax.set_ylabel("energy [$t$]")
    ax.legend(loc="upper center", markerscale=2)

style.save(draw, "wannier_kagome")
