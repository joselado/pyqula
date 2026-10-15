# h2 and W as in the section above
import numpy as np
import style
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction
g = geometry.honeycomb_lattice()
h2 = g.get_hamiltonian()
h2.add_sublattice_imbalance(1.0) # a gapped semiconductor
W = density_interaction(h2,Vr=lambda r1,r2: 0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25))
# nv=nc=2 keeps both members of each spin-degenerate pair
opts = dict(V=W,nq=20,nk=8,nv=2,nc=2,n=4)
qs,es = h2.get_exciton_bands(**opts) # the four lowest excitons along the path
qs0,es0 = h2.get_exciton_bands(kernel="none",**opts) # the bare continuum
x, x0 = (qs+1)/20, (qs0+1)/20 # position along the path, from Gamma through K, M, K to Gamma

def draw(ax, c):
    ax.scatter(x0, es0.real, s=30, color=c.ink2, alpha=0.5,
               label="free pairs (lowest)")
    ax.scatter(x, es.real, s=14, color=c.blue, label="excitons")
    ax.set_xticks([1/3, 1/2, 2/3, 1], ["K", "M", "K'", r"$\Gamma$"])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("center-of-mass momentum $Q$")
    ax.set_ylabel(r"energy [$t$]")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

style.save(draw, "excitons_bands")
