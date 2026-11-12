import numpy as np
import style
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

hc = geometry.chain().get_hamiltonian(has_spin=True) # conduction electrons
h = KondoLatticeHamiltonian(hc)
Jp = 2.0 # J' of the Kondo term J'(c^dag sigma c).S, in units of t
h2 = h.get_mean_field_hamiltonian(J=2*Jp, filling=0.5, nk=200,
        mf=([Jp+0.0j], [0.0])) # half filling, the Kondo insulator
energies = np.linspace(-6., 6., 300)
(ks, es, A) = h2.get_composite_spectral_function(energies=energies, nk=200,
        delta=0.05, hybridization_scale=1.5)
k = 2*np.pi*ks[:,0] # the k-path goes from 0 to 2 pi

def draw(ax, c):
    vmax = A["cc"].max()
    for (a, key, title) in [(ax[0], "cc", "conduction electron, $A_{cc}$"),
                            (ax[1], "FF", "composite fermion, $A_{FF}$")]:
        im = a.pcolormesh(k, es, A[key].T, cmap=c.cmap, vmin=0, vmax=vmax,
                          shading="nearest", rasterized=True)
        style.kpath(a, k, ["0", r"$\pi$", r"$2\pi$"])
        a.set_xlabel("momentum $k$")
        a.set_title(title)
    ax[0].set_ylabel("energy [$t$]")
    style.colorbar(im, ax[1], "spectral function")

style.save(draw, "pseudofermions_composite", npanels=2)
