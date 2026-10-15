import numpy as np
import style
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

gc = geometry.chain()
hc = gc.get_hamiltonian(has_spin=True) # conduction electrons
h = KondoLatticeHamiltonian(hc)

seed = ([0.3+0.0j],[0.0]) # (V,lam), see the caveat below for why
h2 = h.get_mean_field_hamiltonian(J=1.5,filling=0.15,nk=200,mf=seed)
(k,e,f) = h2.get_bands(operator="zposition") # bands, and weight on the f-sites
k = 2*np.pi*k/k.max() # the k-path goes from 0 to 2 pi
(k0,e0) = hc.get_bands() # conduction band alone, as a reference
k0, e0 = 2*np.pi*k0/k0.max(), e0 - h2.fermi # energies from the chemical potential

def draw(ax, c):
    ax.plot(k0, e0, ".", ms=1, color=c.ink2)
    sc = ax.scatter(k, e, c=2*f-1, s=6, cmap=c.diverging, vmin=-1, vmax=1)
    style.colorbar(sc, ax, "conduction (-1) to f-site (+1)")
    ax.axhline(0, color=c.ink, lw=0.8, ls="--") # chemical potential
    style.kpath(ax, k, ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel("momentum $k$")
    ax.set_ylabel("energy [$t$]")

style.save(draw, "pseudofermions_kondo")
